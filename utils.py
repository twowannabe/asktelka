"""Utility functions: time helpers, mood classification, cheap intent, escaping."""

import os
import re
from datetime import datetime

from config import LOCAL_TZ, disabled_chats

# ---------------------- TEXT HELPERS ----------------------
def escape_markdown_v2(text: str) -> str:
    escape_chars = r"_[]()~>#+-=|{}.!\\"
    return re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", text)


def lowercase_first(text: str) -> str:
    if not text:
        return text
    return text[0].lower() + text[1:]


def is_bot_enabled(chat_id: int) -> bool:
    return chat_id not in disabled_chats

# ---------------------- TIME HELPERS ----------------------
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


def list_ogg_files(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    files = []
    for name in os.listdir(folder):
        if name.lower().endswith(".ogg"):
            files.append(os.path.join(folder, name))
    return files

# ---------------------- MOOD CLASSIFICATION ----------------------
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
    ("flirty", [
        r"\bфлирт", r"\bсекси", r"\bгорячо\b", r"\bпоцелуй", r"\bобнима", r"\bласк", r"\bсоблазн",
    ]),
]


def classify_mood(text: str) -> tuple[str | None, str]:
    t = (text or "").lower()
    if not t:
        return None, ""
    for label, patterns in MOOD_PATTERNS:
        for p in patterns:
            if re.search(p, t, flags=re.IGNORECASE):
                note = f"detected:{label}"
                return label, note
    return None, ""


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
