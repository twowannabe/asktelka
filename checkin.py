"""Check-in job: periodic messages to inactive users."""

import os
import random
import asyncio

from telegram.error import TelegramError
from telegram.ext import CallbackContext

from datetime import timedelta

from config import (
    NUDES_DIR, CHECKIN_PHOTO_CHANCE, CHECKIN_PHOTO_CAPTIONS,
    LOCAL_TZ, VOICE_DIR_MORNING, VOICE_DIR_EVENING,
    LISA_MOODS, LISA_MOOD_TIME_WEIGHTS, LEVEL_PERSONALITIES,
    LEVEL_SELFIE_UNLOCK,
    RITUAL_SELFIE_CHANCE, RITUAL_VOICE_CHANCE,
    THOUGHT_CHANCE, THOUGHT_ACTIVE_DAYS,
    SELFIE_CAPTIONS,
    client, logger,
)
from db import (
    get_last_contacts, get_user_settings,
    set_last_checkin_date, update_last_interaction,
    get_user_memory, get_user_level_info, get_lisa_mood, set_lisa_mood,
    get_last_ritual_date, set_last_ritual_date,
    get_last_thought_date, set_last_thought_date,
)
from utils import (
    local_now, local_date_str, start_of_local_day,
    is_morning, is_evening, list_ogg_files, lowercase_first,
)


async def generate_checkin_text(first_name: str, mood_label: str | None = None, user_id: int | None = None) -> str:
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

    memory_ctx = ""
    if user_id:
        memory = get_user_memory(user_id)
        if memory:
            memory_ctx = f" Вот что ты помнишь о пользователе из прошлых разговоров: {memory}"

    name_ctx = f" Пользователя зовут {first_name}. Обращайся по имени." if first_name else ""

    prompt = (
        f"Ты Лиза — тёплая, заботливая девушка с лёгким флиртом. "
        f"Напиши короткое (1-2 предложения) естественное сообщение пользователю.{name_ctx} "
        f"Сейчас {time_of_day}. Ты давно не общалась с этим человеком и хочешь узнать, как у него дела.{mood_ctx}{memory_ctx} "
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
                text = await generate_checkin_text(first_name=first_name, mood_label=mood_label, user_id=int(user_id))

                if chat_type != "private" and username:
                    text = f"@{username}, {text[0].lower()}{text[1:]}"

                await send_checkin_voice_or_text(context.bot, int(chat_id), text)

            update_last_interaction(int(user_id), int(chat_id), first_name)

            logger.info(f"Check-in sent to user {user_id} ({first_name}) in chat {chat_id}")

        except TelegramError as e:
            logger.warning(f"Telegram error for user {user_id}: {e}")
        except Exception as e:
            logger.error(f"Check-in error for user {user_id}: {e}", exc_info=True)


async def generate_ritual_text(first_name: str, ritual_type: str, mood_label: str | None,
                               user_id: int | None, lisa_mood_prompt: str, user_level: int) -> str:
    personality = LEVEL_PERSONALITIES.get(user_level, LEVEL_PERSONALITIES[7])

    if ritual_type == "morning":
        task = "пожелай доброго утра, спроси про планы на день"
    else:
        task = "пожелай спокойной ночи, спроси как прошёл день"

    memory_ctx = ""
    if user_id:
        memory = get_user_memory(user_id)
        if memory:
            memory_ctx = f" Память о пользователе: {memory}"

    mood_ctx = f" Настроение пользователя: {mood_label}." if mood_label else ""
    name_ctx = f" Пользователя зовут {first_name}." if first_name else ""

    prompt = (
        f"{personality} "
        f"Ты Лиза. {task}.{name_ctx}{mood_ctx}{memory_ctx} "
        f"Твоё настроение: {lisa_mood_prompt} "
        "1-2 предложения, как в мессенджере. "
        "ВАЖНО: начинай с маленькой буквы. "
        "ОБЯЗАТЕЛЬНО используй букву «ё» везде, где она нужна. "
        "Никогда не используй ремарки в скобках, звуковые эффекты и ролеплей-действия."
    )

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="grok-3-mini",
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=30,
        )
        text = (response.choices[0].message.content or "").strip()
        if text:
            return lowercase_first(text)
    except Exception as e:
        logger.error(f"Ritual text generation error: {e}", exc_info=True)

    if ritual_type == "morning":
        return f"доброе утро{', ' + first_name if first_name else ''} ☀️ какие планы на сегодня?"
    return f"спокойной ночи{', ' + first_name if first_name else ''} 🌙 как прошёл день?"


async def send_ritual(context: CallbackContext) -> None:
    now = local_now()
    hour = now.hour

    if 9 <= hour < 10:
        ritual_type = "morning"
    elif 22 <= hour < 23:
        ritual_type = "evening"
    else:
        return

    rows = get_last_contacts()
    if not rows:
        return

    today_str = local_date_str()

    for (user_id, chat_id, last_interaction, first_name, username, chat_type) in rows:
        try:
            if chat_type != "private":
                continue

            st = get_user_settings(int(user_id))
            if st.get("do_not_write_first"):
                continue

            if get_last_ritual_date(int(user_id)) == today_str:
                continue

            set_last_ritual_date(int(user_id), today_str)

            level_info = get_user_level_info(int(user_id))
            user_level = level_info["level"]
            mood_label = st.get("mood_label")

            lisa_mood_key = get_lisa_mood()
            lisa_mood_data = LISA_MOODS.get(lisa_mood_key, LISA_MOODS["playful"])

            text = await generate_ritual_text(
                first_name=first_name,
                ritual_type=ritual_type,
                mood_label=mood_label,
                user_id=int(user_id),
                lisa_mood_prompt=lisa_mood_data["prompt_mod"],
                user_level=user_level,
            )

            # Chance for selfie
            if (user_level >= LEVEL_SELFIE_UNLOCK
                    and random.random() < RITUAL_SELFIE_CHANCE * lisa_mood_data.get("selfie_mult", 1.0)):
                try:
                    from gpt import generate_selfie
                    photo_bytes = await generate_selfie()
                    if photo_bytes:
                        import io
                        caption = random.choice(SELFIE_CAPTIONS)
                        await context.bot.send_photo(chat_id=int(chat_id), photo=io.BytesIO(photo_bytes), caption=caption)
                        logger.info(f"Ritual selfie sent to {user_id}")
                        continue
                except Exception as e:
                    logger.error(f"Ritual selfie error for {user_id}: {e}", exc_info=True)

            # Chance for TTS voice
            if random.random() < RITUAL_VOICE_CHANCE * lisa_mood_data.get("voice_mult", 1.0):
                try:
                    from gpt import text_to_voice, get_ogg_duration
                    import io
                    voice_data = await text_to_voice(text)
                    if voice_data:
                        duration = int(get_ogg_duration(voice_data)) or None
                        voice_file = io.BytesIO(voice_data)
                        voice_file.name = "voice.ogg"
                        await context.bot.send_voice(chat_id=int(chat_id), voice=voice_file, duration=duration)
                        logger.info(f"Ritual voice sent to {user_id}")
                        continue
                except Exception as e:
                    logger.error(f"Ritual voice error for {user_id}: {e}", exc_info=True)

            await context.bot.send_message(chat_id=int(chat_id), text=text)
            logger.info(f"Ritual text sent to {user_id} ({ritual_type})")

        except TelegramError as e:
            logger.warning(f"Ritual telegram error for {user_id}: {e}")
        except Exception as e:
            logger.error(f"Ritual error for {user_id}: {e}", exc_info=True)


async def send_lisa_thoughts(context: CallbackContext) -> None:
    now = local_now()
    if now.hour < 10 or now.hour >= 22:
        return

    rows = get_last_contacts()
    if not rows:
        return

    today_str = local_date_str()
    active_cutoff = now - timedelta(days=THOUGHT_ACTIVE_DAYS)

    for (user_id, chat_id, last_interaction, first_name, username, chat_type) in rows:
        try:
            if chat_type != "private":
                continue

            st = get_user_settings(int(user_id))
            if st.get("do_not_write_first"):
                continue

            if get_last_thought_date(int(user_id)) == today_str:
                continue

            # Check if user was active recently
            if not last_interaction:
                continue
            if last_interaction.tzinfo is None:
                last_local = LOCAL_TZ.localize(last_interaction)
            else:
                last_local = last_interaction.astimezone(LOCAL_TZ)
            if last_local < active_cutoff:
                continue

            # 50% chance
            if random.random() > THOUGHT_CHANCE:
                continue

            set_last_thought_date(int(user_id), today_str)

            level_info = get_user_level_info(int(user_id))
            user_level = level_info["level"]
            memory = get_user_memory(int(user_id))

            lisa_mood_key = get_lisa_mood()
            lisa_mood_data = LISA_MOODS.get(lisa_mood_key, LISA_MOODS["playful"])

            from gpt import generate_lisa_thought
            text = await generate_lisa_thought(
                user_name=first_name or "",
                memory=memory,
                user_level=user_level,
                lisa_mood_prompt=lisa_mood_data["prompt_mod"],
            )

            await context.bot.send_message(chat_id=int(chat_id), text=f"💭 {text}")
            logger.info(f"Lisa thought sent to {user_id}")

        except TelegramError as e:
            logger.warning(f"Thought telegram error for {user_id}: {e}")
        except Exception as e:
            logger.error(f"Thought error for {user_id}: {e}", exc_info=True)


async def update_lisa_mood(context: CallbackContext) -> None:
    now = local_now()
    hour = now.hour

    if 6 <= hour < 12:
        period = "morning"
    elif 12 <= hour < 18:
        period = "afternoon"
    elif 18 <= hour < 24:
        period = "evening"
    else:
        period = "night"

    if random.random() < 0.7:
        new_mood = random.choice(LISA_MOOD_TIME_WEIGHTS[period])
    else:
        new_mood = random.choice(list(LISA_MOODS.keys()))

    set_lisa_mood(new_mood)
    logger.info(f"Lisa mood updated: {new_mood} (period={period})")
