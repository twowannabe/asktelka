"""Utility functions: time helpers, mood classification, cheap intent, escaping."""

import os
import re
import random
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


def typing_delay(text: str) -> float:
    """Adaptive typing delay based on text length — shorter texts get shorter pauses."""
    n = len(text)
    if n < 30:
        return random.uniform(0.5, 1.5)
    if n < 80:
        return random.uniform(1.0, 2.5)
    if n < 150:
        return random.uniform(2.0, 4.0)
    return random.uniform(3.0, 5.0)


def split_message(text: str) -> list[str]:
    """Split a long reply into 2-3 natural chunks for more human-like delivery."""
    if len(text) < 60:
        return [text]

    # Try splitting by double newline first
    parts = text.split("\n\n")
    if len(parts) >= 2:
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) >= 2:
            return _merge_short(parts)

    # Try splitting by single newline
    parts = text.split("\n")
    if len(parts) >= 2:
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) >= 2:
            return _merge_short(parts)

    # Try splitting by sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) >= 2:
        return _merge_short(sentences)

    return [text]


def _merge_short(parts: list[str], min_len: int = 15, max_parts: int = 3) -> list[str]:
    """Merge short fragments together, cap at max_parts."""
    merged: list[str] = []
    buf = ""
    for p in parts:
        if buf:
            buf += "\n" + p
        else:
            buf = p
        if len(buf) >= min_len:
            merged.append(buf)
            buf = ""
    if buf:
        if merged:
            merged[-1] += "\n" + buf
        else:
            merged.append(buf)

    # Cap at max_parts by merging trailing parts
    while len(merged) > max_parts:
        merged[-2] += "\n" + merged[-1]
        merged.pop()

    # If only got 1 part after merging, return as-is
    if len(merged) < 2:
        return ["\n".join(parts)] if parts else merged

    return merged


# ---------------------- HUMANIZE ----------------------
# Slang substitutions Lisa would naturally use
_SLANG_MAP = [
    (r"\bсейчас\b", "щас"),
    (r"\bвообще\b", "вобще"),
    (r"\bкажется\b", "кажись"),
    (r"\bнормально\b", "норм"),
    (r"\bхорошо\b", "хорош"),
    (r"\bпожалуйста\b", "пожааалуйста"),
    (r"\bсерьёзно\b", "серьёзн"),
    (r"\bнаверное\b", "наверн"),
    (r"\bкороче\b", "корч"),
    (r"\bчеловек\b", "челик"),
]

# Common typos: doubled letter, swapped adjacent letters, missed letter
def _add_typo(word: str) -> str:
    """Add a single realistic typo to a word."""
    if len(word) < 4:
        return word
    chars = list(word)
    method = random.choice(["double", "swap", "skip"])
    idx = random.randint(1, len(chars) - 2)
    if method == "double":
        chars.insert(idx, chars[idx])
    elif method == "swap" and idx < len(chars) - 1:
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    elif method == "skip":
        chars.pop(idx)
    return "".join(chars)


def humanize_text(text: str) -> str:
    """Post-process GPT reply to add human-like artifacts (slang, typos)."""
    if not text or len(text) < 5:
        return text

    # Slang substitutions — 15% chance per match
    for pattern, replacement in _SLANG_MAP:
        if random.random() < 0.15:
            text = re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)

    # Typo — 5% chance, pick one random word
    if random.random() < 0.05:
        words = text.split()
        if len(words) >= 3:
            # Pick a word that's long enough (not emoji, not short)
            candidates = [i for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
            if candidates:
                idx = random.choice(candidates)
                words[idx] = _add_typo(words[idx])
                text = " ".join(words)

    # Trailing punctuation variation — 8% chance: add extra chars
    if random.random() < 0.08:
        if text.endswith("!"):
            text = text[:-1] + random.choice(["!!", "!!!", "!1"])
        elif text.endswith("?"):
            text = text[:-1] + random.choice(["??", "???"])
        elif text.endswith("."):
            text = text[:-1] + random.choice(["...", "..", ""])

    return text


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
