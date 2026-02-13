"""Check-in job: periodic messages to inactive users."""

import os
import random
import asyncio

from telegram.error import TelegramError
from telegram.ext import CallbackContext

from config import (
    NUDES_DIR, CHECKIN_PHOTO_CHANCE, CHECKIN_PHOTO_CAPTIONS,
    LOCAL_TZ, VOICE_DIR_MORNING, VOICE_DIR_EVENING,
    client, logger,
)
from db import (
    get_last_contacts, get_user_settings,
    set_last_checkin_date, update_last_interaction,
)
from utils import (
    local_now, local_date_str, start_of_local_day,
    is_morning, is_evening, list_ogg_files, lowercase_first,
)


async def generate_checkin_text(first_name: str, mood_label: str | None = None) -> str:
    now = local_now()
    if is_morning(now):
        time_of_day = "утро"
    elif is_evening(now):
        time_of_day = "вечер"
    else:
        time_of_day = "день"

    mood_ctx = ""
    if mood_label:
        mood_ctx = f" Последнее известное настроение пользователя: {mood_label}. Учти это мягко, не акцентируй."

    name_ctx = f" Пользователя зовут {first_name}. Обращайся по имени." if first_name else ""

    prompt = (
        f"Ты Лиза — тёплая, заботливая девушка с лёгким флиртом. "
        f"Напиши короткое (1-2 предложения) естественное сообщение пользователю.{name_ctx} "
        f"Сейчас {time_of_day}. Ты давно не общалась с этим человеком и хочешь узнать, как у него дела.{mood_ctx} "
        f"Не используй шаблонные фразы вроде 'как дела'. Будь живой, уникальной, как настоящая девушка в мессенджере. "
        f"ВАЖНО: начинай сообщение с маленькой буквы, кроме имён собственных. "
        f"Пиши только текст сообщения, без кавычек."
    )

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="grok-3-mini",
                messages=[{"role": "user", "content": prompt}],
                n=1,
            ),
            timeout=30,
        )
        text = (response.choices[0].message.content or "").strip()
        if text:
            return lowercase_first(text)
    except Exception as e:
        logger.error(f"GPT checkin generation error: {e}")

    name_part = f" {first_name}" if first_name else ""
    return f"привет{name_part} 💛 давно не общались, как ты?"


async def send_checkin_voice_or_text(bot, chat_id: int, text: str):
    now = local_now()
    folder = VOICE_DIR_MORNING if is_morning(now) else VOICE_DIR_EVENING if is_evening(now) else ""
    voice_files = list_ogg_files(folder) if folder else []

    if voice_files and random.random() < 0.6:
        path = random.choice(voice_files)
        try:
            with open(path, "rb") as f:
                await bot.send_voice(chat_id=chat_id, voice=f)
            return
        except Exception as e:
            logger.error(f"Failed to send voice {path}: {e}")

    await bot.send_message(chat_id=chat_id, text=text)


async def check_lonely_users(context: CallbackContext) -> None:
    now = local_now()

    if now.hour >= 23 or now.hour < 9:
        return

    rows = get_last_contacts()
    if not rows:
        return

    today_start = start_of_local_day()
    today_str = local_date_str()

    for (user_id, chat_id, last_interaction, first_name, username, chat_type) in rows:
        try:
            st = get_user_settings(int(user_id))

            if st.get("do_not_write_first"):
                continue

            if st.get("last_checkin_date") == today_str:
                continue

            set_last_checkin_date(int(user_id), today_str)

            if not last_interaction:
                continue

            if last_interaction.tzinfo is None:
                last_local = LOCAL_TZ.localize(last_interaction)
            else:
                last_local = last_interaction.astimezone(LOCAL_TZ)

            if last_local >= today_start:
                continue

            photos = [f for f in os.listdir(NUDES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))] if os.path.isdir(NUDES_DIR) else []
            if photos and random.random() < CHECKIN_PHOTO_CHANCE and chat_type == "private":
                photo_path = os.path.join(NUDES_DIR, random.choice(photos))
                caption = random.choice(CHECKIN_PHOTO_CAPTIONS)
                with open(photo_path, "rb") as ph:
                    await context.bot.send_photo(chat_id=int(chat_id), photo=ph, caption=caption)
            else:
                mood_label = st.get("mood_label")
                text = await generate_checkin_text(first_name=first_name, mood_label=mood_label)

                if chat_type != "private" and username:
                    text = f"@{username}, {text[0].lower()}{text[1:]}"

                await send_checkin_voice_or_text(context.bot, int(chat_id), text)

            update_last_interaction(int(user_id), int(chat_id), first_name)

            logger.info(f"Check-in sent to user {user_id} ({first_name}) in chat {chat_id}")

        except TelegramError as e:
            logger.warning(f"Telegram error for user {user_id}: {e}")
        except Exception as e:
            logger.error(f"Check-in error for user {user_id}: {e}", exc_info=True)
