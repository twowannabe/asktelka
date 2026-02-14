"""Command handlers and message handlers."""

import base64
import io
import os
import re
import random
import asyncio
import time
import tempfile
from datetime import datetime, timezone

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from config import (
    LEVELS, LOCAL_TZ,
    NUDES_DIR, NUDES_KEYWORDS, NUDES_THRESHOLD, NUDES_THRESHOLD_BY_LEVEL, NUDES_TEASE_REPLIES, NUDES_SEND_REPLIES,
    EMOJI_REACTION_CHANCE, REACTION_EMOJIS, CHEAP_REACTION_CHANCE, CHEAP_REACTIONS,
    MEDIA_REACTIONS, MEDIA_REACTION_CHANCE,
    RANDOM_GPT_RESPONSE_CHANCE,
    GROUP_COMMENT_CHANCE, GROUP_COMMENT_BUFFER_SIZE, chat_message_buffer,
    JEALOUSY_MIN_LEVEL, JEALOUSY_THRESHOLD, JEALOUSY_CHANCE, JEALOUSY_COOLDOWN_SEC,
    JEALOUSY_REACTIONS, jealousy_counters, jealousy_cooldowns,
    LEVEL_VOICE_UNLOCK, LEVEL_SELFIE_UNLOCK,
    XP_PER_TEXT, XP_PER_VOICE, XP_PER_NUDES, XP_PER_SELFIE,
    SELFIE_CHANCE, SELFIE_CAPTIONS,
    MEMORY_SUMMARIZE_EVERY,
    ACHIEVEMENTS, ACHIEVEMENT_MESSAGES,
    disabled_chats, user_personalities, nudes_request_count, active_games,
    logger,
)
from db import (
    get_db_connection, log_interaction, save_message, load_context, load_group_context, clear_context,
    load_user_personality_from_db, upsert_user_personality,
    update_last_interaction, ensure_user_state_row,
    set_do_not_write_first, get_user_settings, set_cheap_cooldown, set_mood,
    get_user_level_info, get_next_level_xp, add_xp, send_level_up, get_user_voice_chance,
    get_user_memory, save_user_memory, increment_memory_counter,
    get_user_achievements, grant_achievement,
)
from gpt import ask_chatgpt, text_to_voice, transcribe_voice, summarize_memory, generate_chat_comment, generate_jealous_comment, react_to_photo, generate_selfie
from games import handle_game_response
from utils import (
    escape_markdown_v2, lowercase_first, is_bot_enabled,
    classify_mood, cheap_intent, local_now,
)


async def _update_memory(user_id: int):
    try:
        old_summary = get_user_memory(user_id)
        messages = load_context(user_id, limit=30)
        new_summary = await summarize_memory(old_summary, messages)
        save_user_memory(user_id, new_summary)
        logger.info(f"Memory updated for user {user_id}: {new_summary[:80]}...")
    except Exception as e:
        logger.error(f"Memory update error for user {user_id}: {e}", exc_info=True)


# ---------------------- COMMAND HANDLERS ----------------------

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "привет 💛 я Лиза. я тут, чтобы поддерживать тебя, флиртовать и иногда спрашивать, как ты 😏\n"
        "если не хочешь, чтобы я писала первой — набери /dontwritefirst"
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "команды:\n"
        "/start — начать\n"
        "/help — помощь\n"
        "/disable — выключить бота в этом чате\n"
        "/reset — сбросить историю GPT\n"
        "/set_personality <текст> — задать стиль общения (с флиртом)\n"
        "/dontwritefirst — не писать первой (для тебя)\n"
        "/writefirst — снова можно писать первой\n"
        "/mood — показать, что я запомнила про твоё настроение\n"
        "/clear_mood — очистить память настроения\n"
        "/stats — статистика общения с Лизой\n"
        "/level — твой уровень и XP\n"
        "/achievements — твои ачивки\n"
        "/selfie [подсказка] — селфи от Лизы 📸\n\n"
        "🎮 мини-игры:\n"
        "/truth — правда или действие (+2 XP)\n"
        "/guess — угадай число 1-100 (+3 XP)\n"
        "/riddle — загадка от Лизы (+5 XP)\n"
        "/quiz — викторина с кнопками (+4 XP)\n"
    )
    await update.message.reply_text(text)


async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM askgbt_logs WHERE user_id = %s", (user_id,))
        total = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM askgbt_logs WHERE user_id = %s AND gpt_reply LIKE '[voice]%%'", (user_id,))
        voice_replies = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM askgbt_logs WHERE user_id = %s AND user_message LIKE '[voice]%%'", (user_id,))
        voice_sent = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM askgbt_logs WHERE user_id = %s AND gpt_reply LIKE '[nudes]%%'", (user_id,))
        nudes = cur.fetchone()[0]

        cur.execute("SELECT MIN(timestamp), MAX(timestamp) FROM askgbt_logs WHERE user_id = %s", (user_id,))
        first_msg, last_msg = cur.fetchone()

        cur.close()
        conn.close()

        days = (last_msg - first_msg).days + 1 if first_msg and last_msg else 1
        avg = round(total / days, 1) if days > 0 else 0

        text = (
            f"📊 твоя статистика с Лизой:\n\n"
            f"💬 всего сообщений: {total}\n"
            f"🎤 голосовых от тебя: {voice_sent}\n"
            f"🔊 голосовых от Лизы: {voice_replies}\n"
            f"🔞 нюдсов выпросил: {nudes}\n"
            f"📅 дней общения: {days}\n"
            f"📈 в среднем: {avg} сообщ/день"
        )
    except Exception as e:
        logger.error(f"Stats error: {e}", exc_info=True)
        text = "не смогла посчитать статистику 😔"

    await update.message.reply_text(text)


async def level_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    info = get_user_level_info(user_id)
    xp = info["xp"]
    level = info["level"]
    title = info["title"]
    streak = info["streak_days"]

    next_xp = get_next_level_xp(level)
    if next_xp:
        progress = xp - [t for l, t, _ in LEVELS if l == level][0]
        needed = next_xp - [t for l, t, _ in LEVELS if l == level][0]
        pct = min(int(progress / needed * 10), 10) if needed > 0 else 10
        bar = "▓" * pct + "░" * (10 - pct)
        next_line = f"до уровня {level + 1}: [{bar}] {xp}/{next_xp} XP"
    else:
        next_line = "максимальный уровень! 👑"

    streak_line = f"🔥 Стрик: {streak} дн." if streak >= 2 else ""
    streak_bonus = " (x1.5 XP)" if streak >= 2 else ""

    text = (
        f"✨ Уровень {level} — {title}\n"
        f"⭐ {xp} XP\n"
        f"{next_line}\n"
        f"{streak_line}{streak_bonus}"
    ).strip()

    await update.message.reply_text(text)


async def set_personality_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    personality = " ".join(context.args).strip()
    if not personality:
        await update.message.reply_text("напиши так: /set_personality <описание стиля>")
        return
    user_personalities[user_id] = personality
    upsert_user_personality(user_id, personality)
    await update.message.reply_text("принято 💛 теперь общаюсь в этом стиле.")


async def dontwritefirst_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    set_do_not_write_first(user_id, True)
    await update.message.reply_text("поняла 🫶 теперь я не буду писать тебе первой.")


async def writefirst_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    set_do_not_write_first(user_id, False)
    await update.message.reply_text("хорошо 😊 теперь могу иногда начинать разговор сама.")


async def mood_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    st = get_user_settings(user_id)
    if not st["mood_label"]:
        await update.message.reply_text("я пока ничего не запомнила про твоё настроение.")
        return
    when = st["mood_updated_at"].astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M") if st["mood_updated_at"] else "не знаю когда"
    await update.message.reply_text(
        f"я запомнила: настроение **{st['mood_label']}** (обновляла: {when}).",
        parse_mode=ParseMode.MARKDOWN,
    )


async def clear_mood_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    set_mood(user_id, None, "")
    await update.message.reply_text("окей. я очистила память про настроение ✨")


async def disable_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    disabled_chats.add(chat_id)
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO disabled_chats (chat_id) VALUES (%s) ON CONFLICT DO NOTHING", (chat_id,))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"DB disable chat error: {e}", exc_info=True)
    await update.message.reply_text("поняла. я выключилась в этом чате.")


async def is_user_admin(update: Update) -> bool:
    try:
        member = await update.effective_chat.get_member(update.effective_user.id)
        return member.status in ["administrator", "creator"]
    except Exception as e:
        logger.error(f"Admin check error: {e}")
        return False


async def reset_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    clear_context(user_id)
    await update.message.reply_text("окей. я сбросила историю разговора ✨")


# ---------------------- ACHIEVEMENTS ----------------------

async def _send_achievement_notification(bot, chat_id: int, key: str):
    ach = ACHIEVEMENTS[key]
    msg = random.choice(ACHIEVEMENT_MESSAGES).format(title=ach["title"], emoji=ach["emoji"])
    try:
        await bot.send_message(chat_id=chat_id, text=msg)
    except Exception as e:
        logger.error(f"Achievement notification error: {e}")


async def _check_and_grant(bot, chat_id: int, user_id: int, key: str):
    if grant_achievement(user_id, key):
        await _send_achievement_notification(bot, chat_id, key)


async def achievements_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    earned = get_user_achievements(user_id)
    lines = []
    for key, ach in ACHIEVEMENTS.items():
        if key in earned:
            lines.append(f"✅ {ach['emoji']} {ach['title']} — {ach['desc']}")
        else:
            lines.append(f"⬜ {ach['emoji']} {ach['title']} — {ach['desc']}")
    text = "🏆 твои ачивки:\n\n" + "\n".join(lines)
    await update.message.reply_text(text)


# ---------------------- SELFIE ----------------------

async def selfie_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    user_level_info = get_user_level_info(user_id)
    user_level = user_level_info["level"]

    if user_level < LEVEL_SELFIE_UNLOCK:
        await update.message.reply_text(
            f"селфи доступны с уровня {LEVEL_SELFIE_UNLOCK} 😏 пока пообщаемся?"
        )
        return

    hint = " ".join(context.args).strip() if context.args else ""

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_PHOTO)
    photo_bytes = await generate_selfie(hint)
    if not photo_bytes:
        await update.message.reply_text("не получилось сделать селфи 😔 попробуй позже")
        return

    caption = random.choice(SELFIE_CAPTIONS)
    await context.bot.send_photo(chat_id=chat_id, photo=io.BytesIO(photo_bytes), caption=caption)
    log_interaction(user_id, update.effective_user.username or "", f"/selfie {hint}".strip(), f"[selfie] {caption}")

    _, new_level, leveled_up = add_xp(user_id, XP_PER_SELFIE)
    if leveled_up:
        await send_level_up(context.bot, chat_id, new_level, update.effective_chat.type)


# ---------------------- MESSAGE HANDLERS ----------------------

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    chat = update.effective_chat
    user = update.effective_user
    chat_id = chat.id
    user_id = user.id
    user_username = user.username or ""

    if chat.type != "private" and not is_bot_enabled(chat_id):
        return

    user_first_name = user.first_name or user_username or ""
    update_last_interaction(user_id, chat_id, user_first_name, user_username, chat.type)
    ensure_user_state_row(user_id)

    # Achievement: night_owl
    if 0 <= local_now().hour < 5:
        await _check_and_grant(context.bot, chat_id, user_id, "night_owl")

    voice = update.message.voice or update.message.audio
    if not voice:
        return

    try:
        file = await context.bot.get_file(voice.file_id)
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp_path = tmp.name
            await file.download_to_drive(tmp_path)

        text = await transcribe_voice(tmp_path)
        os.unlink(tmp_path)

        if not text:
            await update.message.reply_text("не расслышала, скажи ещё раз 🎧")
            return

        logger.info(f"Voice transcribed: {text[:100]}...")

    except Exception as e:
        logger.error(f"Voice transcription error: {e}", exc_info=True)
        await update.message.reply_text("не смогла разобрать голосовое 😔")
        return

    mood_label, mood_note = classify_mood(text)
    if mood_label:
        set_mood(user_id, mood_label, mood_note)

    user_level_info = get_user_level_info(user_id)
    user_level = user_level_info["level"]

    personality = user_personalities.get(user_id) or load_user_personality_from_db(user_id) or ""
    st = get_user_settings(user_id)
    user_mood = st.get("mood_label") or ""

    is_group = chat.type != "private"

    save_message(user_id, "user", text, chat_id=chat_id, sender_name=user_first_name)

    count = increment_memory_counter(user_id)
    if count >= MEMORY_SUMMARIZE_EVERY:
        asyncio.create_task(_update_memory(user_id))

    memory = get_user_memory(user_id)
    if is_group:
        messages = load_group_context(chat_id, limit=20)
    else:
        messages = load_context(user_id, limit=10)

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    await asyncio.sleep(random.uniform(1, 2))

    reply = await ask_chatgpt(
        messages,
        user_name=user_first_name,
        personality=personality,
        mood_label=user_mood,
        memory=memory,
        user_level=user_level,
        is_group=is_group,
    )

    if not reply.strip():
        reply = "ммм… напиши ещё 😅"

    save_message(user_id, "assistant", reply, chat_id=chat_id, sender_name="Лиза")

    reply_to_message_id = update.message.message_id

    sent_as_voice = False
    voice_data = await text_to_voice(reply) if user_level >= LEVEL_VOICE_UNLOCK else None
    if voice_data:
        try:
            if chat.type == "private":
                await context.bot.send_voice(chat_id=chat_id, voice=io.BytesIO(voice_data))
            else:
                await context.bot.send_voice(chat_id=chat_id, voice=io.BytesIO(voice_data), reply_to_message_id=reply_to_message_id)
            sent_as_voice = True
        except Exception as e:
            logger.error(f"Voice send error: {e}", exc_info=True)

    if not sent_as_voice:
        try:
            if chat.type == "private":
                await context.bot.send_message(chat_id=chat_id, text=reply)
            else:
                await update.message.reply_text(reply, reply_to_message_id=reply_to_message_id)
        except Exception as e:
            logger.error(f"Telegram send error: {e}", exc_info=True)

    log_interaction(user_id, user_username, f"[voice] {text}", f"{'[voice] ' if sent_as_voice else ''}{reply}")

    _, new_level, leveled_up = add_xp(user_id, XP_PER_VOICE)
    if leveled_up:
        await send_level_up(context.bot, chat_id, new_level, chat.type)

    # Achievement: streak_7
    if get_user_level_info(user_id)["streak_days"] >= 7:
        await _check_and_grant(context.bot, chat_id, user_id, "streak_7")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    chat = update.effective_chat
    user = update.effective_user

    chat_id = chat.id
    user_id = user.id
    user_username = user.username or ""

    text = (update.message.text or "").strip()
    if not text:
        return

    if chat.type != "private" and not is_bot_enabled(chat_id):
        return

    user_first_name = user.first_name or user_username or ""

    update_last_interaction(user_id, chat_id, user_first_name, user_username, chat.type)
    ensure_user_state_row(user_id)

    # Achievement: night_owl
    if 0 <= local_now().hour < 5:
        await _check_and_grant(context.bot, chat_id, user_id, "night_owl")

    mood_label, mood_note = classify_mood(text)
    if mood_label:
        set_mood(user_id, mood_label, mood_note)

    if user_id in active_games and active_games[user_id]["type"] != "quiz":
        handled = await handle_game_response(user_id, text, update, context)
        if handled:
            return

    # Nudes request detection
    user_level_info = get_user_level_info(user_id)
    user_level = user_level_info["level"]

    text_lower = text.lower()
    is_nudes_request = any(kw in text_lower for kw in NUDES_KEYWORDS)
    if is_nudes_request and (chat.type == "private" or (update.message.reply_to_message and update.message.reply_to_message.from_user and update.message.reply_to_message.from_user.id == context.bot.id)):
        nudes_threshold = NUDES_THRESHOLD_BY_LEVEL.get(user_level, NUDES_THRESHOLD)
        nudes_request_count[user_id] += 1
        if nudes_request_count[user_id] >= nudes_threshold:
            photos = [f for f in os.listdir(NUDES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))] if os.path.isdir(NUDES_DIR) else []
            if photos:
                photo_path = os.path.join(NUDES_DIR, random.choice(photos))
                caption = random.choice(NUDES_SEND_REPLIES)
                try:
                    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
                    await asyncio.sleep(random.uniform(2, 5))
                    with open(photo_path, "rb") as ph:
                        await context.bot.send_photo(chat_id=chat_id, photo=ph, caption=caption)
                    nudes_request_count[user_id] = 0
                    log_interaction(user_id, user_username, text, f"[nudes] {caption}")
                    # Achievement: first_nudes
                    await _check_and_grant(context.bot, chat_id, user_id, "first_nudes")
                    _, new_level, leveled_up = add_xp(user_id, XP_PER_NUDES)
                    if leveled_up:
                        await send_level_up(context.bot, chat_id, new_level, chat.type)
                    return
                except Exception as e:
                    logger.error(f"Nudes send error: {e}", exc_info=True)
        else:
            tease = random.choice(NUDES_TEASE_REPLIES)
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            await asyncio.sleep(random.uniform(1, 3))
            if chat.type == "private":
                await context.bot.send_message(chat_id=chat_id, text=tease)
            else:
                await update.message.reply_text(tease, reply_to_message_id=update.message.message_id)
            log_interaction(user_id, user_username, text, f"[tease] {tease}")
            return

    bot_username = context.bot.username
    is_bot_mentioned = f"@{bot_username}".lower() in text.lower()
    is_reply = update.message.reply_to_message is not None
    is_reply_to_bot = is_reply and update.message.reply_to_message.from_user and update.message.reply_to_message.from_user.id == context.bot.id

    should_gpt = False
    text_to_process = ""
    reply_to_message_id = update.message.message_id
    reply_context = ""  # text of the message being replied to

    if chat.type == "private":
        should_gpt = True
        text_to_process = text

    elif is_bot_mentioned and not is_reply:
        should_gpt = True
        text_to_process = re.sub(rf"@{re.escape(bot_username)}", "", text, flags=re.IGNORECASE).strip()

    elif is_reply_to_bot:
        should_gpt = True
        original_text = update.message.reply_to_message.text or update.message.reply_to_message.caption or ""
        if original_text.strip():
            reply_context = original_text.strip()
        text_to_process = text

    elif is_reply and is_bot_mentioned:
        should_gpt = True
        original = update.message.reply_to_message.text or update.message.reply_to_message.caption or ""
        if original.strip():
            reply_context = original.strip()
        text_to_process = re.sub(rf"@{re.escape(bot_username)}", "", text, flags=re.IGNORECASE).strip() or text

    elif random.random() < RANDOM_GPT_RESPONSE_CHANCE:
        should_gpt = True
        text_to_process = text

    if not should_gpt:
        # Group chat commentary buffer
        buf = chat_message_buffer[chat_id]
        buf.append(f"{user_first_name}: {text}")
        if len(buf) > GROUP_COMMENT_BUFFER_SIZE:
            del buf[: len(buf) - GROUP_COMMENT_BUFFER_SIZE]

        # Jealousy mechanic for high-level users ignoring Lisa
        jealousy_counters[chat_id][user_id] += 1
        if (
            user_level >= JEALOUSY_MIN_LEVEL
            and jealousy_counters[chat_id][user_id] >= JEALOUSY_THRESHOLD
            and time.time() - jealousy_cooldowns[chat_id][user_id] >= JEALOUSY_COOLDOWN_SEC
            and random.random() < JEALOUSY_CHANCE
        ):
            try:
                await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
                await asyncio.sleep(random.uniform(1, 3))
                comment = await generate_jealous_comment(list(buf), user_first_name, user_level)
                if not comment:
                    comment = random.choice(JEALOUSY_REACTIONS)
                await context.bot.send_message(chat_id=chat_id, text=comment)
            except Exception as e:
                logger.error(f"Jealousy comment error: {e}", exc_info=True)
            jealousy_counters[chat_id][user_id] = 0
            jealousy_cooldowns[chat_id][user_id] = time.time()
            return

        if random.random() < GROUP_COMMENT_CHANCE:
            try:
                await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
                await asyncio.sleep(random.uniform(1, 3))
                comment = await generate_chat_comment(list(buf))
                if comment:
                    await context.bot.send_message(chat_id=chat_id, text=comment)
                    buf.clear()
                    return
            except Exception as e:
                logger.error(f"Group comment send error: {e}", exc_info=True)

        if random.random() < EMOJI_REACTION_CHANCE:
            emoji = random.choice(REACTION_EMOJIS)
            try:
                from telegram import ReactionTypeEmoji
                await update.message.set_reaction([ReactionTypeEmoji(emoji=emoji)])
                return
            except Exception as e:
                logger.debug(f"Emoji reaction failed: {e}")

        st = get_user_settings(user_id)
        now_epoch = int(datetime.now(timezone.utc).timestamp())
        if now_epoch < int(st["cheap_reaction_cooldown_until"] or 0):
            return

        if random.random() < CHEAP_REACTION_CHANCE:
            intent = cheap_intent(text)
            reply = random.choice(CHEAP_REACTIONS.get(intent, CHEAP_REACTIONS["fallback"]))
            if user_first_name and random.random() < 0.5:
                reply = f"{user_first_name}, {reply[0].lower()}{reply[1:]}"
            try:
                await update.message.reply_text(reply, reply_to_message_id=reply_to_message_id)
                cooldown = now_epoch + random.randint(10 * 60, 25 * 60)
                set_cheap_cooldown(user_id, cooldown)
                log_interaction(user_id, user_username, text, f"[cheap]{reply}")
            except Exception as e:
                logger.error(f"Cheap reply send error: {e}")
        return

    # GPT path — user addressed the bot, reset jealousy counter
    jealousy_counters[chat_id][user_id] = 0

    if not text_to_process:
        text_to_process = text

    # Prepend reply context to the message
    if reply_context:
        text_to_process = f'[в ответ на: "{reply_context}"] {text_to_process}'

    is_group = chat.type != "private"

    personality = user_personalities.get(user_id) or load_user_personality_from_db(user_id) or ""
    st = get_user_settings(user_id)
    user_mood = st.get("mood_label") or ""

    save_message(user_id, "user", text_to_process, chat_id=chat_id, sender_name=user_first_name)

    count = increment_memory_counter(user_id)
    if count >= MEMORY_SUMMARIZE_EVERY:
        asyncio.create_task(_update_memory(user_id))

    memory = get_user_memory(user_id)
    if is_group:
        messages = load_group_context(chat_id, limit=20)
    else:
        messages = load_context(user_id, limit=10)

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    await asyncio.sleep(random.uniform(1, 4))

    reply = await ask_chatgpt(
        messages,
        user_name=user_first_name,
        personality=personality,
        mood_label=user_mood,
        memory=memory,
        user_level=user_level,
        is_group=is_group,
    )

    if not reply.strip():
        reply = "ммм… напиши ещё 😅"

    save_message(user_id, "assistant", reply, chat_id=chat_id, sender_name="Лиза")

    # Detect requested voice style
    voice_style = ""
    force_voice = False
    if re.search(r"стон|застон|постон", text_lower):
        voice_style = "moan"
        force_voice = True
    elif re.search(r"шепн|шёпот|шепот|пошепч|шепч", text_lower):
        voice_style = "whisper"
        force_voice = True
    elif re.search(r"озвуч|голос|скажи это|произнеси|повтори голосом|скажи вслух", text_lower):
        force_voice = True

    sent_as_voice = False
    voice_chance = get_user_voice_chance(user_id)
    if force_voice or (user_level >= LEVEL_VOICE_UNLOCK and random.random() < voice_chance):
        voice_data = await text_to_voice(reply, style=voice_style)
        if voice_data:
            try:
                if chat.type == "private":
                    await context.bot.send_voice(chat_id=chat_id, voice=io.BytesIO(voice_data))
                else:
                    await context.bot.send_voice(chat_id=chat_id, voice=io.BytesIO(voice_data), reply_to_message_id=reply_to_message_id)
                sent_as_voice = True
            except Exception as e:
                logger.error(f"Voice send error: {e}", exc_info=True)

    if not sent_as_voice:
        try:
            if chat.type == "private":
                await context.bot.send_message(chat_id=chat_id, text=reply)
            else:
                escaped = escape_markdown_v2(reply)
                if len(escaped) > 4096:
                    escaped = escaped[:4096]
                await update.message.reply_text(
                    escaped,
                    parse_mode=ParseMode.MARKDOWN_V2,
                    reply_to_message_id=reply_to_message_id,
                )
        except BadRequest:
            await context.bot.send_message(chat_id=chat_id, text=reply)
        except Exception as e:
            logger.error(f"Telegram send error: {e}", exc_info=True)

    log_interaction(user_id, user_username, text_to_process, f"{'[voice] ' if sent_as_voice else ''}{reply}")

    _, new_level, leveled_up = add_xp(user_id, XP_PER_TEXT)
    if leveled_up:
        await send_level_up(context.bot, chat_id, new_level, chat.type)

    # Achievement: msg_100
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM askgbt_logs WHERE user_id = %s", (user_id,))
        msg_count = cur.fetchone()[0]
        cur.close()
        conn.close()
        if msg_count >= 100:
            await _check_and_grant(context.bot, chat_id, user_id, "msg_100")
    except Exception as e:
        logger.error(f"Achievement msg_100 check error: {e}", exc_info=True)

    # Achievement: streak_7
    if get_user_level_info(user_id)["streak_days"] >= 7:
        await _check_and_grant(context.bot, chat_id, user_id, "streak_7")

    # Spontaneous selfie (private chat only, level gated)
    if chat.type == "private" and user_level >= LEVEL_SELFIE_UNLOCK and random.random() < SELFIE_CHANCE:
        try:
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_PHOTO)
            photo_bytes = await generate_selfie()
            if photo_bytes:
                caption = random.choice(SELFIE_CAPTIONS)
                await context.bot.send_photo(chat_id=chat_id, photo=io.BytesIO(photo_bytes), caption=caption)
        except Exception as e:
            logger.error(f"Spontaneous selfie error: {e}", exc_info=True)


async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    chat = update.effective_chat
    user = update.effective_user
    chat_id = chat.id
    user_id = user.id

    if chat.type != "private" and not is_bot_enabled(chat_id):
        return

    if chat.type != "private" and random.random() > MEDIA_REACTION_CHANCE:
        return

    # Try vision analysis for photos
    photo = update.message.photo
    if photo:
        try:
            user_level_info = get_user_level_info(user_id)
            user_level = user_level_info["level"]

            file = await context.bot.get_file(photo[-1].file_id)
            photo_bytes = await file.download_as_bytearray()
            image_base64 = base64.b64encode(photo_bytes).decode("utf-8")

            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            reply = await react_to_photo(image_base64, user_level)

            if reply:
                await asyncio.sleep(random.uniform(1, 3))
                if chat.type == "private":
                    await context.bot.send_message(chat_id=chat_id, text=reply)
                else:
                    await update.message.reply_text(reply, reply_to_message_id=update.message.message_id)
                return
        except Exception as e:
            logger.error(f"Vision reaction error: {e}", exc_info=True)

    # Fallback: emoji reaction or random text
    if random.random() < 0.4:
        try:
            from telegram import ReactionTypeEmoji
            emoji = random.choice(REACTION_EMOJIS)
            await update.message.set_reaction([ReactionTypeEmoji(emoji=emoji)])
            return
        except Exception:
            pass

    reply = random.choice(MEDIA_REACTIONS)
    await asyncio.sleep(random.uniform(1, 3))
    try:
        if chat.type == "private":
            await context.bot.send_message(chat_id=chat_id, text=reply)
        else:
            await update.message.reply_text(reply, reply_to_message_id=update.message.message_id)
    except Exception as e:
        logger.error(f"Media reaction error: {e}")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Exception while handling an update:", exc_info=context.error)
    if isinstance(update, Update) and update.message:
        try:
            await update.message.reply_text("ой… у меня что-то пошло не так 😅")
        except Exception:
            pass
