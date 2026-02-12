#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Companion Telegram bot ("милая девушка") with:
- daily check-ins if user hasn't interacted today
- different morning/evening check-in texts
- optional voice check-ins ("как ты?") from local .ogg files
- mood memory (simple heuristic classification) stored in DB
- "do not write first" mode per user
- cheap rule-based reactions without GPT (fast/cheap)
- GPT replies when mentioned / replied-to / random chance

Tested conceptually for python-telegram-bot v20+ (async).
"""

import os
import re
import random
import logging
import asyncio
from collections import defaultdict
from datetime import datetime, timezone

import psycopg2
from decouple import config
from openai import AsyncOpenAI
import openai

import pytz
from telegram import Update
from telegram.constants import ParseMode
from telegram.error import BadRequest, TelegramError
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    CallbackContext,
    filters,
)

# ---------------------- CONFIG ----------------------
TELEGRAM_TOKEN = config("TELEGRAM_TOKEN")
OPENAI_API_KEY = config("OPENAI_API_KEY")

DB_HOST = config("DB_HOST")
DB_PORT = config("DB_PORT")
DB_NAME = config("DB_NAME")
DB_USER = config("DB_USER")
DB_PASSWORD = config("DB_PASSWORD")

# Optional: RSS URL from your old bot (keep if needed)
NEWS_RSS_URL = config("NEWS_RSS_URL", default="")

# Timezone for "today / morning / evening"
LOCAL_TZ = pytz.timezone("Europe/Podgorica")

# Probability of random GPT reply (when not mentioned/replied)
RANDOM_GPT_RESPONSE_CHANCE = 1 / 50  # Increased for more flirty interactions
# Probability of cheap reaction (rule-based) on any message (when not GPT-triggered)
CHEAP_REACTION_CHANCE = 1 / 12

# How often to scan for "lonely users"
CHECK_LONELY_INTERVAL_SEC = 60 * 60 * 3  # every 3 hours

# Dumb mode: shorter, simpler replies
DUMB_MODE = True
MAX_WORDS = 15
# Consider "not talked today" if last_interaction is before local-day start
# (We implement day start logic directly, not a fixed hour interval)

# Voice folders (put your .ogg here)
# Example structure:
#   voices/checkin_morning/*.ogg
#   voices/checkin_evening/*.ogg
VOICE_DIR_MORNING = os.path.join(os.path.dirname(__file__), "voices", "checkin_morning")
VOICE_DIR_EVENING = os.path.join(os.path.dirname(__file__), "voices", "checkin_evening")

# ---------------------- OPENAI ----------------------
openai.api_key = OPENAI_API_KEY
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# ---------------------- LOGGING ----------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("companion_bot")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)

# ---------------------- RUNTIME STATE ----------------------
conversation_context = defaultdict(list)   # user_id -> OpenAI messages
group_status = defaultdict(bool)          # chat_id -> enabled/disabled
user_personalities = defaultdict(str)     # user_id -> personality override

default_personality = (
    "Ты Лиза. Ты тёплая, заботливая, немного застенчивая девушка с ноткой соблазна. "
    "Ты поддерживаешь, сочувствуешь, задаёшь простые вопросы, но добавляешь лёгкий флирт: комплименты, намёки на близость, игривые эмодзи. "
    "Ты не умничаешь, не споришь, не давишь, не манипулируешь. Твой стиль — мягкий, живой, человечный, с сексуальным подтекстом. Пиши коротко."
)

async def set_personality_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    personality = " ".join(context.args).strip()
    if not personality:
        await update.message.reply_text("Напиши так: /set_personality <описание стиля>")
        return
    user_personalities[user_id] = personality
    upsert_user_personality(user_id, personality)
    await update.message.reply_text("Принято 💛 Теперь общаюсь в этом стиле.")

async def dontwritefirst_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    set_do_not_write_first(user_id, True)
    await update.message.reply_text("Поняла 🫶 Теперь я не буду писать тебе первой.")

async def writefirst_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    set_do_not_write_first(user_id, False)
    await update.message.reply_text("Хорошо 😊 Теперь могу иногда начинать разговор сама.")

async def mood_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    st = get_user_settings(user_id)
    if not st["mood_label"]:
        await update.message.reply_text("Я пока ничего не запомнила про твоё настроение.")
        return
    when = st["mood_updated_at"].astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M") if st["mood_updated_at"] else "не знаю когда"
    await update.message.reply_text(
        f"Я запомнила: настроение **{st['mood_label']}** (обновляла: {when}).",
        parse_mode=ParseMode.MARKDOWN,
    )

async def clear_mood_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    set_mood(user_id, None, "")
    await update.message.reply_text("Окей. Я очистила память про настроение ✨")

async def enable_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if await is_user_admin(update):
        group_status[chat_id] = True
        await update.message.reply_text("Окей 😊 Я включилась в этом чате.")
    else:
        await update.message.reply_text("Эту команду могут использовать только админы.")

async def disable_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if await is_user_admin(update):
        group_status[chat_id] = False
        await update.message.reply_text("Поняла. Я выключилась в этом чате.")
    else:
        await update.message.reply_text("Эту команду могут использовать только админы.")

async def is_user_admin(update: Update) -> bool:
    try:
        member = await update.effective_chat.get_member(update.effective_user.id)
        return member.status in ["administrator", "creator"]
    except Exception as e:
        logger.error(f"Admin check error: {e}")
        return False

async def reset_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    conversation_context[user_id] = []           # очищаем контекст GPT
    await update.message.reply_text("Окей. Я сбросила историю разговора ✨")
        
# ---------------------- DB HELPERS ----------------------
# (Оставил без изменений, чтобы не трогать БД)

def get_db_connection():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )

def init_db():
    """
    Tables:
    - askgbt_logs: interaction logs
    - user_personalities: per-user personality
    - user_last_contact: per-user last interaction timestamp + last known chat_id
    - user_state: mood memory + do_not_write_first + checkin_sent_date + cheap cooldowns
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS askgbt_logs (
            id SERIAL PRIMARY KEY,
            user_id BIGINT,
            user_username TEXT,
            user_message TEXT,
            gpt_reply TEXT,
            timestamp TIMESTAMP
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS user_personalities (
            user_id BIGINT PRIMARY KEY,
            personality TEXT
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS user_last_contact (
            user_id BIGINT PRIMARY KEY,
            chat_id BIGINT,
            last_interaction TIMESTAMP,
            first_name TEXT
        )
        """)

        # Migration: add first_name column if missing
        cur.execute("""
            ALTER TABLE user_last_contact ADD COLUMN IF NOT EXISTS first_name TEXT
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS user_state (
            user_id BIGINT PRIMARY KEY,

            -- "mood memory"
            mood_label TEXT,
            mood_note TEXT,
            mood_updated_at TIMESTAMP,

            -- settings
            do_not_write_first BOOLEAN DEFAULT FALSE,

            -- anti-spam: last date when we sent check-in (local date string YYYY-MM-DD)
            last_checkin_date TEXT,

            -- anti-spam: cheap reaction cooldown (epoch seconds)
            cheap_reaction_cooldown_until BIGINT DEFAULT 0
        )
        """)

        conn.commit()
        cur.close()
        conn.close()
        logger.info("DB initialized")
    except Exception as e:
        logger.error(f"DB init error: {e}", exc_info=True)

def log_interaction(user_id, user_username, user_message, gpt_reply):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        ts = datetime.now()
        cur.execute("""
            INSERT INTO askgbt_logs (user_id, user_username, user_message, gpt_reply, timestamp)
            VALUES (%s, %s, %s, %s, %s)
        """, (user_id, user_username, user_message, gpt_reply, ts))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"DB log_interaction error: {e}", exc_info=True)

def load_user_personality_from_db(user_id: int) -> str | None:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT personality FROM user_personalities WHERE user_id=%s", (user_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        return row[0] if row else None
    except Exception as e:
        logger.error(f"DB load personality error: {e}")
        return None

def upsert_user_personality(user_id: int, personality: str):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO user_personalities (user_id, personality)
            VALUES (%s, %s)
            ON CONFLICT (user_id) DO UPDATE SET personality = EXCLUDED.personality
        """, (user_id, personality))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"DB upsert personality error: {e}", exc_info=True)

def update_last_interaction(user_id: int, chat_id: int, first_name: str = ""):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO user_last_contact (user_id, chat_id, last_interaction, first_name)
            VALUES (%s, %s, NOW(), %s)
            ON CONFLICT (user_id) DO UPDATE SET
                last_interaction = NOW(),
                chat_id = EXCLUDED.chat_id,
                first_name = EXCLUDED.first_name
        """, (user_id, chat_id, first_name))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"DB update last interaction error: {e}", exc_info=True)

def ensure_user_state_row(user_id: int):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO user_state (user_id)
            VALUES (%s)
            ON CONFLICT (user_id) DO NOTHING
        """, (user_id,))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"DB ensure user_state error: {e}", exc_info=True)

def set_do_not_write_first(user_id: int, value: bool):
    ensure_user_state_row(user_id)
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            UPDATE user_state
            SET do_not_write_first = %s
            WHERE user_id = %s
        """, (value, user_id))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"DB set do_not_write_first error: {e}", exc_info=True)

def get_user_settings(user_id: int) -> dict:
    ensure_user_state_row(user_id)
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT do_not_write_first, last_checkin_date, cheap_reaction_cooldown_until,
                   mood_label, mood_note, mood_updated_at
            FROM user_state
            WHERE user_id=%s
        """, (user_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row:
            return {
                "do_not_write_first": False,
                "last_checkin_date": None,
                "cheap_reaction_cooldown_until": 0,
                "mood_label": None,
                "mood_note": None,
                "mood_updated_at": None,
            }

        return {
            "do_not_write_first": bool(row[0]),
            "last_checkin_date": row[1],
            "cheap_reaction_cooldown_until": int(row[2] or 0),
            "mood_label": row[3],
            "mood_note": row[4],
            "mood_updated_at": row[5],
        }
    except Exception as e:
        logger.error(f"DB get settings error: {e}", exc_info=True)
        return {
            "do_not_write_first": False,
            "last_checkin_date": None,
            "cheap_reaction_cooldown_until": 0,
            "mood_label": None,
            "mood_note": None,
            "mood_updated_at": None,
        }

def set_last_checkin_date(user_id: int, date_str: str):
    ensure_user_state_row(user_id)
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            UPDATE user_state
            SET last_checkin_date = %s
            WHERE user_id = %s
        """, (date_str, user_id))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"DB set last_checkin_date error: {e}", exc_info=True)

def set_cheap_cooldown(user_id: int, until_epoch: int):
    ensure_user_state_row(user_id)
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            UPDATE user_state
            SET cheap_reaction_cooldown_until = %s
            WHERE user_id = %s
        """, (until_epoch, user_id))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"DB set cheap cooldown error: {e}", exc_info=True)

def set_mood(user_id: int, mood_label: str, mood_note: str):
    ensure_user_state_row(user_id)
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            UPDATE user_state
            SET mood_label=%s, mood_note=%s, mood_updated_at=NOW()
            WHERE user_id=%s
        """, (mood_label, mood_note, user_id))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"DB set mood error: {e}", exc_info=True)

def get_last_contacts() -> list[tuple[int, int, datetime, str]]:
    """
    Returns list of (user_id, chat_id, last_interaction, first_name)
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT user_id, chat_id, last_interaction, COALESCE(first_name, '') FROM user_last_contact")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"DB get last contacts error: {e}", exc_info=True)
        return []

# ---------------------- UTIL ----------------------
def escape_markdown_v2(text: str) -> str:
    escape_chars = r"_[]()~>#+-=|{}.!\\"
    return re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", text)

def is_bot_enabled(chat_id: int) -> bool:
    return group_status.get(chat_id, False)

def local_now() -> datetime:
    return datetime.now(LOCAL_TZ)

def local_date_str(dt: datetime | None = None) -> str:
    dt = dt or local_now()
    return dt.strftime("%Y-%m-%d")

def start_of_local_day(dt: datetime | None = None) -> datetime:
    dt = dt or local_now()
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)

def is_morning(dt: datetime | None = None) -> bool:
    dt = dt or local_now()
    return 6 <= dt.hour < 12

def is_evening(dt: datetime | None = None) -> bool:
    dt = dt or local_now()
    return 18 <= dt.hour < 23

async def generate_checkin_text(first_name: str, mood_label: str | None = None) -> str:
    """
    Generate a natural, unique check-in message via GPT instead of templates.
    """
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
        f"Пиши только текст сообщения, без кавычек."
    )

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="gpt-5-nano",
                messages=[{"role": "user", "content": prompt}],
                n=1,
            ),
            timeout=30,
        )
        text = (response.choices[0].message.content or "").strip()
        if text:
            return text
    except Exception as e:
        logger.error(f"GPT checkin generation error: {e}")

    # Fallback if GPT fails
    name_part = f" {first_name}" if first_name else ""
    return f"Привет{name_part} 💛 Давно не общались, как ты?"

def list_ogg_files(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    files = []
    for name in os.listdir(folder):
        if name.lower().endswith(".ogg"):
            files.append(os.path.join(folder, name))
    return files

async def send_checkin_voice_or_text(bot, chat_id: int, text: str):
    """
    If voice files exist for the current time bucket, send a random voice.
    If not, fallback to text.
    """
    now = local_now()
    folder = VOICE_DIR_MORNING if is_morning(now) else VOICE_DIR_EVENING if is_evening(now) else ""
    voice_files = list_ogg_files(folder) if folder else []

    # 50/50: sometimes voice, sometimes text (if voice exists)
    if voice_files and random.random() < 0.6:
        path = random.choice(voice_files)
        try:
            with open(path, "rb") as f:
                await bot.send_voice(chat_id=chat_id, voice=f)
            return
        except Exception as e:
            logger.error(f"Failed to send voice {path}: {e}")

    await bot.send_message(chat_id=chat_id, text=text)

# ---------------------- MOOD MEMORY (CHEAP HEURISTICS) ----------------------
MOOD_PATTERNS = [
    ("sad", [
        r"\bгрустн", r"\bпечаль", r"\bплохо\b", r"\bхреново\b", r"\bуныл", r"\bодиноко\b",
        r"\bнет сил\b", r"\bплачу\b", r"\bслёзы\b", r"\bдепресс",
    ]),
    ("anxious", [
        r"\bтревог", r"\bпаник", r"\bстрашно\b", r"\bбоюсь\b", r"\bне могу успоко",
        r"\bсердце\b.*\bколот", r"\bдрож", r"\bнакрывает\b",
    ]),
    ("happy", [
        r"\bкласс\b", r"\bсупер\b", r"\bотлично\b", r"\bрад\b", r"\bсчастлив", r"\bкайф\b",
        r"\bура\b", r"\bкруто\b",
    ]),
    ("angry", [
        r"\bзлюсь\b", r"\bбесит\b", r"\bраздраж", r"\bненавиж",
    ]),
    ("tired", [
        r"\bустал\b", r"\bустала\b", r"\bсонн", r"\bвымот", r"\bнет энергии\b",
    ]),
    ("flirty", [  # Added for flirty detection
        r"\bфлирт", r"\bсекси", r"\bгорячо\b", r"\bпоцелуй", r"\bобнима", r"\bласк", r"\bсоблазн",
    ]),
]

def classify_mood(text: str) -> tuple[str | None, str]:
    """
    Returns (mood_label or None, mood_note short).
    """
    t = (text or "").lower()
    if not t:
        return None, ""
    for label, patterns in MOOD_PATTERNS:
        for p in patterns:
            if re.search(p, t, flags=re.IGNORECASE):
                note = f"detected:{label}"
                return label, note
    return None, ""

# ---------------------- CHEAP REACTIONS (NO GPT) ----------------------
CHEAP_REACTIONS = {
    "greeting": [
        "Приветик 💛 Ты выглядишь так заманчиво сегодня 😏",
        "Ой, привет 😊 Я уже скучала... по твоему вниманию ❤️",
        "Привет! Я тут 🫶 Хочу услышать, что у тебя на уме 🔥",
    ],
    "thanks": [
        "Пожалуйста 💛 А ты можешь отблагодарить поцелуем? 😘",
        "Всегда пожалуйста 😊 Я люблю, когда ты говоришь 'спасибо' — это сексуально 😉",
        "Я рядом 🫶 Может, обнимемся? 🫂",
    ],
    "sad": [
        "Ох… обниму тебя мысленно 🫂 и поцелую в щёчку. Расскажи, что случилось? 😘",
        "Мне жаль, что тебе так… Я рядом. Давай я отвлеку тебя флиртом? 🔥",
    ],
    "anxious": [
        "Слушай… давай чуть-чуть медленно вдох–выдох? Я с тобой, и мои объятия теплые ❤️",
        "Я рядом. Если хочешь, расскажи, что именно тревожит... или давай пофлиртуем, чтобы отвлечься 😉",
    ],
    "happy": [
        "Ура 🥰 Я рада за тебя! Ты такой вдохновляющий, когда счастлив 🔥",
        "Ой, кайф! Расскажешь подробнее? 😊 Я бы добавила комплимент 😏",
    ],
    "flirty": [  # Added for flirty reactions
        "Ого, ты флиртуешь? Мне нравится 😘 Продолжай 🔥",
        "Ты такой соблазнительный... Я готова ответить взаимностью ❤️",
    ],
    "fallback": [
        "Я тебя слышу 💛 И хочу услышать больше... интимного 😉",
        "Мм… понимаю 🫶 Ты меня заводишь своими словами 🔥",
        "Я рядом. Может, пофлиртуем? 😏",
    ],
}

def cheap_intent(text: str) -> str:
    t = (text or "").lower().strip()
    if re.search(r"\b(привет|здаров|хай|hello|доброе утро|добрый вечер)\b", t):
        return "greeting"
    if re.search(r"\b(спасибо|благодарю|thx|thanks)\b", t):
        return "thanks"
    mood, _ = classify_mood(t)
    if mood in ("sad", "anxious", "happy", "flirty"):
        return mood
    return "fallback"

# ---------------------- OPENAI ----------------------
async def ask_chatgpt(messages, user_name: str = "", personality: str = "", mood_label: str = "", dumb_mode: bool = DUMB_MODE) -> str:
    """
    GPT call with optional dumb mode.
    Hard word limit enforced.
    """

    try:
        name_part = f" Пользователя зовут {user_name}. Обращайся по имени." if user_name else ""
        mood_part = f" (У пользователя сейчас настроение: {mood_label}. Учти это мягко.)" if mood_label else ""

        # System prompt injection (only once)
        if dumb_mode:
            system_prompt = (
                "Ты Лиза. "
                "Ты немного глуповатая, наивная и милая девушка. "
                "Отвечай максимум 10 словами. "
                "Используй очень простые слова. "
                "Без умных объяснений. "
                "Без длинных предложений. "
                "Можно говорить 'эээ', 'ну', 'ммм'. "
                "Иногда путай мысли. "
                "Всегда коротко."
                f"{name_part}{mood_part}"
            )
        else:
            base = personality or default_personality
            system_prompt = f"{base}{name_part}{mood_part}"

        # Ensure system role exists at top
        if not messages or messages[0]["role"] != "system":
            messages = [{"role": "system", "content": system_prompt}] + messages

        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="gpt-5-nano",
                messages=messages,
                max_completion_tokens=25 if dumb_mode else 200,
                n=1,
            ),
            timeout=60,
        )

        reply = (response.choices[0].message.content or "").strip()

        if not reply:
            return "эээ… я задумалась 😅"

        # HARD WORD LIMIT
        if dumb_mode:
            words = reply.split()
            reply = " ".join(words[:MAX_WORDS])

        return reply

    except Exception as e:
        logger.error(f"OpenAI error: {e}", exc_info=True)
        return "эээ… я зависла 😳"
    
# ---------------------- COMMANDS ----------------------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет 💛 Я Лиза. Я тут, чтобы поддерживать тебя, флиртовать и иногда спрашивать, как ты. 😏\n"
        "Если не хочешь, чтобы я писала первой — набери /dontwritefirst"
    )

# (Остальные команды без изменений, кроме /help, где добавил упоминание флирта)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "Команды:\n"
        "/start — начать\n"
        "/help — помощь\n"
        "/enable — включить бота в группе (админ)\n"
        "/disable — выключить бота в группе (админ)\n"
        "/reset — сбросить историю GPT\n"
        "/set_personality <текст> — задать стиль общения (с флиртом)\n"
        "/dontwritefirst — не писать первой (для тебя)\n"
        "/writefirst — снова можно писать первой\n"
        "/mood — показать, что я запомнила про твоё настроение\n"
        "/clear_mood — очистить память настроения\n"
    )
    await update.message.reply_text(text)

# (Остальные команды без изменений)

# ---------------------- MESSAGE HANDLER ----------------------
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

    # If group chat: require enable
    if chat.type != "private" and not is_bot_enabled(chat_id):
        return

    user_first_name = user.first_name or user_username or ""

    # Update last interaction (so we don't check-in if they are active)
    update_last_interaction(user_id, chat_id, user_first_name)
    ensure_user_state_row(user_id)

    # Update mood memory on any message (cheap heuristic)
    mood_label, mood_note = classify_mood(text)
    if mood_label:
        set_mood(user_id, mood_label, mood_note)

    bot_username = context.bot.username
    is_bot_mentioned = f"@{bot_username}".lower() in text.lower()
    is_reply = update.message.reply_to_message is not None
    is_reply_to_bot = is_reply and update.message.reply_to_message.from_user and update.message.reply_to_message.from_user.id == context.bot.id

    # Decide response mode
    should_gpt = False
    text_to_process = ""
    reply_to_message_id = update.message.message_id

    # 1) Mention
    if is_bot_mentioned and not is_reply:
        should_gpt = True
        text_to_process = re.sub(rf"@{re.escape(bot_username)}", "", text, flags=re.IGNORECASE).strip()

    # 2) Reply to bot
    elif is_reply_to_bot:
        should_gpt = True
        text_to_process = text

    # 3) Reply + mention (use original replied text if exists)
    elif is_reply and is_bot_mentioned:
        original = update.message.reply_to_message.text or update.message.reply_to_message.caption or ""
        if original.strip():
            should_gpt = True
            text_to_process = original.strip()
        else:
            await update.message.reply_text("Я не вижу текста в исходном сообщении 😔", reply_to_message_id=reply_to_message_id)
            return

    # 4) Random GPT
    elif random.random() < RANDOM_GPT_RESPONSE_CHANCE:
        should_gpt = True
        text_to_process = text

    # If not GPT, maybe cheap reaction
    if not should_gpt:
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
                # set cooldown ~ 10-25 minutes so it doesn't spam
                cooldown = now_epoch + random.randint(10 * 60, 25 * 60)
                set_cheap_cooldown(user_id, cooldown)
                log_interaction(user_id, user_username, text, f"[cheap]{reply}")
            except Exception as e:
                logger.error(f"Cheap reply send error: {e}")
        return

    # GPT path
    if not text_to_process:
        text_to_process = text

    # Load user personality and mood for GPT context
    personality = user_personalities.get(user_id) or load_user_personality_from_db(user_id) or ""
    st = get_user_settings(user_id)
    user_mood = st.get("mood_label") or ""

    conversation_context[user_id].append({
        "role": "user",
        "content": text_to_process
    })

    # Keep last 10 messages only
    conversation_context[user_id] = conversation_context[user_id][-10:]

    reply = await ask_chatgpt(
        conversation_context[user_id],
        user_name=user_first_name,
        personality=personality,
        mood_label=user_mood,
    )

    if not reply.strip():
        reply = "ммм… напиши ещё 😅"

    conversation_context[user_id].append({
        "role": "assistant",
        "content": reply
    })
    conversation_context[user_id] = conversation_context[user_id][-10:]

    try:
        # Safer: MarkdownV2 escape; fallback to plain if fails
        escaped = escape_markdown_v2(reply)
        if len(escaped) > 4096:
            escaped = escaped[:4096]
        await update.message.reply_text(
            escaped,
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_to_message_id=reply_to_message_id,
        )
    except BadRequest:
        await update.message.reply_text(reply, reply_to_message_id=reply_to_message_id)
    except Exception as e:
        logger.error(f"Telegram send error: {e}", exc_info=True)

    log_interaction(user_id, user_username, text_to_process, reply)

# ---------------------- DAILY CHECK-IN JOB ----------------------
# (Оставил без изменений, кроме вызова pick_checkin_text, который уже флиртующий)

async def check_lonely_users(context: CallbackContext) -> None:
    """
    For each user, if:
    - it's not quiet hours (23:00–9:00)
    - do_not_write_first == False
    - we didn't send a check-in today
    - last_interaction is before start of today's local day
    Then generate a natural check-in via GPT and send it.
    """
    now = local_now()

    # Quiet hours: don't bother people at night
    if now.hour >= 23 or now.hour < 9:
        return

    rows = get_last_contacts()
    if not rows:
        return

    today_start = start_of_local_day()
    today_str = local_date_str()

    for (user_id, chat_id, last_interaction, first_name) in rows:
        try:
            st = get_user_settings(int(user_id))

            # Respect "не писать первым"
            if st.get("do_not_write_first"):
                continue

            # If already sent today, skip
            if st.get("last_checkin_date") == today_str:
                continue

            # Mark sent BEFORE sending to prevent duplicates on concurrent runs
            set_last_checkin_date(int(user_id), today_str)

            # If last_interaction is None, skip
            if not last_interaction:
                continue

            # last_interaction is naive timestamp (likely local server time).
            if last_interaction.tzinfo is None:
                last_local = LOCAL_TZ.localize(last_interaction)
            else:
                last_local = last_interaction.astimezone(LOCAL_TZ)

            if last_local >= today_start:
                continue  # they talked today

            # Generate a natural message via GPT
            mood_label = st.get("mood_label")
            text = await generate_checkin_text(first_name=first_name, mood_label=mood_label)

            await send_checkin_voice_or_text(context.bot, int(chat_id), text)

            # Refresh last_interaction so we won't re-ping too soon
            update_last_interaction(int(user_id), int(chat_id), first_name)

            logger.info(f"Check-in sent to user {user_id} ({first_name}) in chat {chat_id}")

        except TelegramError as e:
            logger.warning(f"Telegram error for user {user_id}: {e}")
        except Exception as e:
            logger.error(f"Check-in error for user {user_id}: {e}", exc_info=True)

# ---------------------- ERROR HANDLER ----------------------
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Exception while handling an update:", exc_info=context.error)
    if isinstance(update, Update) and update.message:
        try:
            await update.message.reply_text("Ой… у меня что-то пошло не так 😅")
        except Exception:
            pass

# ---------------------- MAIN ----------------------
def main():
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).read_timeout(60).build()

    init_db()

    application.add_handler(CommandHandler("start", start_cmd))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("enable", enable_cmd))
    application.add_handler(CommandHandler("disable", disable_cmd))
    application.add_handler(CommandHandler("reset", reset_cmd))
    application.add_handler(CommandHandler("set_personality", set_personality_cmd))

    application.add_handler(CommandHandler("dontwritefirst", dontwritefirst_cmd))
    application.add_handler(CommandHandler("writefirst", writefirst_cmd))
    application.add_handler(CommandHandler("mood", mood_cmd))
    application.add_handler(CommandHandler("clear_mood", clear_mood_cmd))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(error_handler)

    # Schedule periodic lonely-user checks
    job_queue = application.job_queue
    job_queue.run_repeating(
        check_lonely_users,
        interval=CHECK_LONELY_INTERVAL_SEC,
        first=60,
    )

    logger.info("Bot started")
    application.run_polling()

if __name__ == "__main__":
    main()