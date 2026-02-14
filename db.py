"""Database helpers: connection, init, all CRUD functions, level system."""

import random
from datetime import datetime

import psycopg2
from psycopg2 import pool

from config import (
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD,
    LEVELS, LEVEL_UP_MESSAGES,
    XP_STREAK_MULTIPLIER, VOICE_CHANCE_BY_LEVEL,
    disabled_chats, logger,
)
from utils import local_date_str, local_now

_connection_pool = None


def _get_pool():
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
        )
    return _connection_pool


def get_db_connection():
    return _get_pool().getconn()


def release_db_connection(conn):
    try:
        _get_pool().putconn(conn)
    except Exception:
        pass


async def run_sync(func, *args, **kwargs):
    """Run a blocking DB function in a thread pool to avoid blocking the event loop."""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


def init_db():
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

        cur.execute("""
        CREATE TABLE IF NOT EXISTS disabled_chats (
            chat_id BIGINT PRIMARY KEY
        )
        """)

        # Migrations
        cur.execute("ALTER TABLE user_last_contact ADD COLUMN IF NOT EXISTS first_name TEXT")
        cur.execute("ALTER TABLE user_last_contact ADD COLUMN IF NOT EXISTS username TEXT")
        cur.execute("ALTER TABLE user_last_contact ADD COLUMN IF NOT EXISTS chat_type TEXT")

        # Load disabled chats into memory
        cur.execute("SELECT chat_id FROM disabled_chats")
        for row in cur.fetchall():
            disabled_chats.add(int(row[0]))

        cur.execute("""
        CREATE TABLE IF NOT EXISTS user_state (
            user_id BIGINT PRIMARY KEY,
            mood_label TEXT,
            mood_note TEXT,
            mood_updated_at TIMESTAMP,
            do_not_write_first BOOLEAN DEFAULT FALSE,
            last_checkin_date TEXT,
            cheap_reaction_cooldown_until BIGINT DEFAULT 0,
            xp INTEGER DEFAULT 0,
            level INTEGER DEFAULT 1,
            streak_days INTEGER DEFAULT 0,
            last_xp_date TEXT
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS conversation_history (
            id SERIAL PRIMARY KEY,
            user_id BIGINT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_conv_user ON conversation_history(user_id, created_at DESC)")

        # Group conversation migrations
        cur.execute("ALTER TABLE conversation_history ADD COLUMN IF NOT EXISTS chat_id BIGINT")
        cur.execute("ALTER TABLE conversation_history ADD COLUMN IF NOT EXISTS sender_name TEXT DEFAULT ''")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_conv_chat ON conversation_history(chat_id, created_at DESC)")

        # Level system migrations
        cur.execute("ALTER TABLE user_state ADD COLUMN IF NOT EXISTS xp INTEGER DEFAULT 0")
        cur.execute("ALTER TABLE user_state ADD COLUMN IF NOT EXISTS level INTEGER DEFAULT 1")
        cur.execute("ALTER TABLE user_state ADD COLUMN IF NOT EXISTS streak_days INTEGER DEFAULT 0")
        cur.execute("ALTER TABLE user_state ADD COLUMN IF NOT EXISTS last_xp_date TEXT")

        cur.execute("""
        CREATE TABLE IF NOT EXISTS user_achievements (
            user_id BIGINT NOT NULL,
            achievement_key TEXT NOT NULL,
            earned_at TIMESTAMP DEFAULT NOW(),
            PRIMARY KEY (user_id, achievement_key)
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS user_memory (
            user_id BIGINT PRIMARY KEY,
            summary TEXT DEFAULT '',
            message_count INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT NOW()
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS lisa_state (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TIMESTAMP DEFAULT NOW()
        )
        """)

        # Horoscope migrations
        cur.execute("ALTER TABLE user_state ADD COLUMN IF NOT EXISTS zodiac_sign TEXT")
        cur.execute("ALTER TABLE user_state ADD COLUMN IF NOT EXISTS last_horoscope_date TEXT")

        # Ritual & thought migrations
        cur.execute("ALTER TABLE user_state ADD COLUMN IF NOT EXISTS last_ritual_date TEXT")
        cur.execute("ALTER TABLE user_state ADD COLUMN IF NOT EXISTS last_thought_date TEXT")

        conn.commit()
        cur.close()
        release_db_connection(conn)
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
        release_db_connection(conn)
    except Exception as e:
        logger.error(f"DB log_interaction error: {e}", exc_info=True)


def save_message(user_id: int, role: str, content: str, chat_id: int = None, sender_name: str = ""):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO conversation_history (user_id, role, content, chat_id, sender_name) VALUES (%s, %s, %s, %s, %s)",
            (user_id, role, content, chat_id, sender_name),
        )
        conn.commit()
        cur.close()
        release_db_connection(conn)
    except Exception as e:
        logger.error(f"DB save_message error: {e}", exc_info=True)


def load_context(user_id: int, limit: int = 10) -> list[dict]:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT role, content FROM conversation_history WHERE user_id=%s ORDER BY created_at DESC LIMIT %s",
            (user_id, limit),
        )
        rows = cur.fetchall()
        cur.close()
        release_db_connection(conn)
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
    except Exception as e:
        logger.error(f"DB load_context error: {e}", exc_info=True)
        return []


def load_group_context(chat_id: int, limit: int = 20) -> list[dict]:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT role, content, sender_name FROM conversation_history WHERE chat_id=%s ORDER BY created_at DESC LIMIT %s",
            (chat_id, limit),
        )
        rows = cur.fetchall()
        cur.close()
        release_db_connection(conn)
        result = []
        for role, content, sender_name in reversed(rows):
            if role == "user" and sender_name:
                result.append({"role": "user", "content": f"{sender_name}: {content}"})
            else:
                result.append({"role": role, "content": content})
        return result
    except Exception as e:
        logger.error(f"DB load_group_context error: {e}", exc_info=True)
        return []


def clear_context(user_id: int):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM conversation_history WHERE user_id=%s", (user_id,))
        conn.commit()
        cur.close()
        release_db_connection(conn)
    except Exception as e:
        logger.error(f"DB clear_context error: {e}", exc_info=True)


def load_user_personality_from_db(user_id: int) -> str | None:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT personality FROM user_personalities WHERE user_id=%s", (user_id,))
        row = cur.fetchone()
        cur.close()
        release_db_connection(conn)
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
        release_db_connection(conn)
    except Exception as e:
        logger.error(f"DB upsert personality error: {e}", exc_info=True)


def update_last_interaction(user_id: int, chat_id: int, first_name: str = "", username: str = "", chat_type: str = ""):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO user_last_contact (user_id, chat_id, last_interaction, first_name, username, chat_type)
            VALUES (%s, %s, NOW(), %s, %s, %s)
            ON CONFLICT (user_id) DO UPDATE SET
                last_interaction = NOW(),
                chat_id = EXCLUDED.chat_id,
                first_name = EXCLUDED.first_name,
                username = EXCLUDED.username,
                chat_type = EXCLUDED.chat_type
        """, (user_id, chat_id, first_name, username, chat_type))
        conn.commit()
        cur.close()
        release_db_connection(conn)
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
        release_db_connection(conn)
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
        release_db_connection(conn)
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
        release_db_connection(conn)

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
        release_db_connection(conn)
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
        release_db_connection(conn)
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
        release_db_connection(conn)
    except Exception as e:
        logger.error(f"DB set mood error: {e}", exc_info=True)


def get_level_for_xp(xp: int) -> tuple[int, str]:
    result = LEVELS[0]
    for lvl, threshold, title in LEVELS:
        if xp >= threshold:
            result = (lvl, title)
        else:
            break
    return result


def get_next_level_xp(current_level: int) -> int | None:
    for lvl, threshold, _ in LEVELS:
        if lvl == current_level + 1:
            return threshold
    return None


def get_user_level_info(user_id: int) -> dict:
    ensure_user_state_row(user_id)
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT COALESCE(xp, 0), COALESCE(level, 1),
                   COALESCE(streak_days, 0), last_xp_date
            FROM user_state WHERE user_id=%s
        """, (user_id,))
        row = cur.fetchone()
        cur.close()
        release_db_connection(conn)
        if row:
            lvl, title = get_level_for_xp(row[0])
            return {"xp": row[0], "level": lvl, "streak_days": row[2],
                    "last_xp_date": row[3], "title": title}
    except Exception as e:
        logger.error(f"DB get_user_level_info error: {e}", exc_info=True)
    return {"xp": 0, "level": 1, "streak_days": 0, "last_xp_date": None, "title": "Незнакомец"}


def add_xp(user_id: int, base_xp: int) -> tuple[int, int, bool]:
    ensure_user_state_row(user_id)
    today = local_date_str()
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT COALESCE(xp, 0), COALESCE(level, 1),
                   COALESCE(streak_days, 0), last_xp_date
            FROM user_state WHERE user_id=%s
        """, (user_id,))
        row = cur.fetchone()
        old_xp, old_level, streak, last_date = row if row else (0, 1, 0, None)

        if last_date == today:
            pass
        else:
            from datetime import timedelta
            yesterday = local_date_str(local_now() - timedelta(days=1))
            if last_date == yesterday:
                streak += 1
            else:
                streak = 1

        xp_gain = base_xp
        if streak >= 2:
            xp_gain = int(base_xp * XP_STREAK_MULTIPLIER)

        new_xp = old_xp + xp_gain
        new_level, new_title = get_level_for_xp(new_xp)
        leveled_up = new_level > old_level

        cur.execute("""
            UPDATE user_state
            SET xp=%s, level=%s, streak_days=%s, last_xp_date=%s
            WHERE user_id=%s
        """, (new_xp, new_level, streak, today, user_id))
        conn.commit()
        cur.close()
        release_db_connection(conn)
        return new_xp, new_level, leveled_up
    except Exception as e:
        logger.error(f"DB add_xp error: {e}", exc_info=True)
        return 0, 1, False


async def send_level_up(bot, chat_id: int, level: int, chat_type: str = "private"):
    _, title = get_level_for_xp(get_next_level_xp(level - 1) or 0)
    for lvl, _, t in LEVELS:
        if lvl == level:
            title = t
            break
    msg = random.choice(LEVEL_UP_MESSAGES).format(title=title, level=level)
    try:
        await bot.send_message(chat_id=chat_id, text=msg)
    except Exception as e:
        logger.error(f"Level-up msg error: {e}")


def get_user_voice_chance(user_id: int) -> float:
    info = get_user_level_info(user_id)
    return VOICE_CHANCE_BY_LEVEL.get(info["level"], 0.2)


def get_user_memory(user_id: int) -> str:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT summary FROM user_memory WHERE user_id=%s", (user_id,))
        row = cur.fetchone()
        cur.close()
        release_db_connection(conn)
        return row[0] if row else ""
    except Exception as e:
        logger.error(f"DB get_user_memory error: {e}", exc_info=True)
        return ""


def save_user_memory(user_id: int, summary: str):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO user_memory (user_id, summary, message_count, updated_at)
            VALUES (%s, %s, 0, NOW())
            ON CONFLICT (user_id) DO UPDATE SET
                summary = EXCLUDED.summary,
                message_count = 0,
                updated_at = NOW()
        """, (user_id, summary))
        conn.commit()
        cur.close()
        release_db_connection(conn)
    except Exception as e:
        logger.error(f"DB save_user_memory error: {e}", exc_info=True)


def increment_memory_counter(user_id: int) -> int:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO user_memory (user_id, message_count)
            VALUES (%s, 1)
            ON CONFLICT (user_id) DO UPDATE SET
                message_count = user_memory.message_count + 1
            RETURNING message_count
        """, (user_id,))
        count = cur.fetchone()[0]
        conn.commit()
        cur.close()
        release_db_connection(conn)
        return count
    except Exception as e:
        logger.error(f"DB increment_memory_counter error: {e}", exc_info=True)
        return 0


def get_user_achievements(user_id: int) -> list[str]:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT achievement_key FROM user_achievements WHERE user_id=%s", (user_id,))
        keys = [row[0] for row in cur.fetchall()]
        cur.close()
        release_db_connection(conn)
        return keys
    except Exception as e:
        logger.error(f"DB get_user_achievements error: {e}", exc_info=True)
        return []


def grant_achievement(user_id: int, key: str) -> bool:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO user_achievements (user_id, achievement_key) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (user_id, key),
        )
        is_new = cur.rowcount > 0
        conn.commit()
        cur.close()
        release_db_connection(conn)
        return is_new
    except Exception as e:
        logger.error(f"DB grant_achievement error: {e}", exc_info=True)
        return False


def get_user_zodiac(user_id: int) -> str | None:
    ensure_user_state_row(user_id)
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT zodiac_sign FROM user_state WHERE user_id=%s", (user_id,))
        row = cur.fetchone()
        cur.close()
        release_db_connection(conn)
        return row[0] if row else None
    except Exception as e:
        logger.error(f"DB get_user_zodiac error: {e}", exc_info=True)
        return None


def set_user_zodiac(user_id: int, sign: str):
    ensure_user_state_row(user_id)
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("UPDATE user_state SET zodiac_sign=%s WHERE user_id=%s", (sign, user_id))
        conn.commit()
        cur.close()
        release_db_connection(conn)
    except Exception as e:
        logger.error(f"DB set_user_zodiac error: {e}", exc_info=True)


def get_last_horoscope_date(user_id: int) -> str | None:
    ensure_user_state_row(user_id)
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT last_horoscope_date FROM user_state WHERE user_id=%s", (user_id,))
        row = cur.fetchone()
        cur.close()
        release_db_connection(conn)
        return row[0] if row else None
    except Exception as e:
        logger.error(f"DB get_last_horoscope_date error: {e}", exc_info=True)
        return None


def set_last_horoscope_date(user_id: int, date_str: str):
    ensure_user_state_row(user_id)
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("UPDATE user_state SET last_horoscope_date=%s WHERE user_id=%s", (date_str, user_id))
        conn.commit()
        cur.close()
        release_db_connection(conn)
    except Exception as e:
        logger.error(f"DB set_last_horoscope_date error: {e}", exc_info=True)


def get_lisa_mood() -> str:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT value FROM lisa_state WHERE key='mood'")
        row = cur.fetchone()
        cur.close()
        release_db_connection(conn)
        return row[0] if row else "playful"
    except Exception as e:
        logger.error(f"DB get_lisa_mood error: {e}", exc_info=True)
        return "playful"


def set_lisa_mood(mood: str):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO lisa_state (key, value, updated_at)
            VALUES ('mood', %s, NOW())
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
        """, (mood,))
        conn.commit()
        cur.close()
        release_db_connection(conn)
    except Exception as e:
        logger.error(f"DB set_lisa_mood error: {e}", exc_info=True)


def get_last_ritual_date(user_id: int) -> str | None:
    ensure_user_state_row(user_id)
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT last_ritual_date FROM user_state WHERE user_id=%s", (user_id,))
        row = cur.fetchone()
        cur.close()
        release_db_connection(conn)
        return row[0] if row else None
    except Exception as e:
        logger.error(f"DB get_last_ritual_date error: {e}", exc_info=True)
        return None


def set_last_ritual_date(user_id: int, date_str: str):
    ensure_user_state_row(user_id)
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("UPDATE user_state SET last_ritual_date=%s WHERE user_id=%s", (date_str, user_id))
        conn.commit()
        cur.close()
        release_db_connection(conn)
    except Exception as e:
        logger.error(f"DB set_last_ritual_date error: {e}", exc_info=True)


def get_last_thought_date(user_id: int) -> str | None:
    ensure_user_state_row(user_id)
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT last_thought_date FROM user_state WHERE user_id=%s", (user_id,))
        row = cur.fetchone()
        cur.close()
        release_db_connection(conn)
        return row[0] if row else None
    except Exception as e:
        logger.error(f"DB get_last_thought_date error: {e}", exc_info=True)
        return None


def set_last_thought_date(user_id: int, date_str: str):
    ensure_user_state_row(user_id)
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("UPDATE user_state SET last_thought_date=%s WHERE user_id=%s", (date_str, user_id))
        conn.commit()
        cur.close()
        release_db_connection(conn)
    except Exception as e:
        logger.error(f"DB set_last_thought_date error: {e}", exc_info=True)


def get_top_users(limit: int = 10) -> list[dict]:
    """Return top users by XP with their names."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT s.user_id, COALESCE(s.xp, 0) AS xp, COALESCE(s.level, 1) AS level,
                   COALESCE(s.streak_days, 0) AS streak,
                   COALESCE(c.first_name, ''), COALESCE(c.username, '')
            FROM user_state s
            LEFT JOIN user_last_contact c ON s.user_id = c.user_id
            WHERE COALESCE(s.xp, 0) > 0
            ORDER BY xp DESC
            LIMIT %s
        """, (limit,))
        rows = cur.fetchall()
        cur.close()
        release_db_connection(conn)
        result = []
        for row in rows:
            lvl, title = get_level_for_xp(row[1])
            result.append({
                "user_id": row[0], "xp": row[1], "level": lvl,
                "streak": row[3], "title": title,
                "first_name": row[4], "username": row[5],
            })
        return result
    except Exception as e:
        logger.error(f"DB get_top_users error: {e}", exc_info=True)
        return []


def get_last_contacts() -> list[tuple]:
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT user_id, chat_id, last_interaction, COALESCE(first_name, ''), COALESCE(username, ''), COALESCE(chat_type, 'private') FROM user_last_contact")
        rows = cur.fetchall()
        cur.close()
        release_db_connection(conn)
        return rows
    except Exception as e:
        logger.error(f"DB get last contacts error: {e}", exc_info=True)
        return []
