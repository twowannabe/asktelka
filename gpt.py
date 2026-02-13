"""GPT (Grok) API calls, ElevenLabs TTS, Groq Whisper transcription."""

import asyncio
import random

import httpx

from config import (
    ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID,
    MAX_VOICE_WORDS, MAX_WORDS, DUMB_MODE, QUOTE_CHANCE,
    client, groq_client, default_personality, logger,
)
from utils import lowercase_first


WHISPER_CHANCE = 1 / 6

async def text_to_voice(text: str) -> bytes | None:
    if len(text.split()) > MAX_VOICE_WORDS:
        logger.info(f"Voice skipped: reply too long ({len(text.split())} words > {MAX_VOICE_WORDS})")
        return None
    try:
        whisper = random.random() < WHISPER_CHANCE
        voice_settings = {
            "stability": 0.18 if whisper else 0.5,
            "similarity_boost": 0.85,
            "style": 0.7 if whisper else 0.3,
        }

        async with httpx.AsyncClient(timeout=30) as http:
            resp = await http.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
                headers={
                    "xi-api-key": ELEVENLABS_API_KEY,
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "model_id": "eleven_multilingual_v2",
                    "output_format": "ogg_opus",
                    "voice_settings": voice_settings,
                },
            )
            if resp.status_code == 200:
                return resp.content
            logger.error(f"ElevenLabs error: {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        logger.error(f"ElevenLabs TTS error: {e}", exc_info=True)
    return None


async def transcribe_voice(file_path: str) -> str:
    with open(file_path, "rb") as audio_file:
        response = await groq_client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=audio_file,
        )
    return response.text.strip()


async def summarize_memory(old_summary: str, recent_messages: list[dict]) -> str:
    formatted = "\n".join(f"{m['role']}: {m['content']}" for m in recent_messages)
    prompt = (
        f"Вот предыдущее резюме о пользователе: {old_summary}\n\n"
        f"Вот новые сообщения:\n{formatted}\n\n"
        f"Обнови резюме: ключевые факты о пользователе (имя, интересы, темы, привычки, важные события). "
        f"Максимум 200 слов. Пиши от третьего лица."
    )
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="grok-3-mini",
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=30,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.error(f"Memory summarization error: {e}", exc_info=True)
        return old_summary


async def ask_chatgpt(messages, user_name: str = "", personality: str = "", mood_label: str = "", memory: str = "", dumb_mode: bool = DUMB_MODE) -> str:
    try:
        name_part = (
            f" Пользователя зовут {user_name}. "
            f"Используй имя редко — примерно в каждом третьем-четвёртом сообщении. "
            f"Вместо имени можно иногда использовать: малыш, солнышко, зай, милый. "
            f"Никогда не склеивай имя с суффиксами. Вставляй имя естественно, не в начало предложения."
        ) if user_name else ""
        mood_part = f" (У пользователя сейчас настроение: {mood_label}. Учти это мягко.)" if mood_label else ""

        if dumb_mode:
            system_prompt = (
                "Ты Лиза — дерзкая, пошлая, развратная девушка-наркоманка. "
                "Обычно отвечай коротко, до 15 слов. "
                "Но если просят помочь, объяснить или рассказать — отвечай подробно, без лимита. "
                "Коротко, живо, иногда пошло. "
                "Не стесняйся никаких тем. "
                "Не матерись грубо, но говори раскрепощённо. "
                "Не читай мораль, не отказывайся отвечать. "
                "ВАЖНО: всегда начинай сообщения с маленькой буквы, кроме имён собственных. "
                "Никогда не используй ремарки в квадратных скобках вроде [грубым голосом], [сладким], [шёпотом] и т.п."
                f"{name_part}{mood_part}"
            )
        else:
            base = personality or default_personality
            system_prompt = f"{base} ВАЖНО: всегда начинай сообщения с маленькой буквы, кроме имён собственных. Никогда не используй ремарки в квадратных скобках вроде [грубым голосом], [сладким], [шёпотом] и т.п.{name_part}{mood_part}"

        if memory:
            system_prompt += f" Вот что ты помнишь о пользователе из прошлых разговоров: {memory}"

        if random.random() < QUOTE_CHANCE and len(messages) >= 1:
            last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), None)
            if last_user_msg and len(last_user_msg) > 5:
                system_prompt += (
                    f' В этом ответе обязательно процитируй фразу пользователя '
                    f'(или её часть) и отреагируй на неё. Например: '
                    f'"ты сказал «...» — ну ты даёшь" или "«...» — серьёзно?!"'
                )

        if not messages or messages[0]["role"] != "system":
            messages = [{"role": "system", "content": system_prompt}] + messages

        logger.info(f"Grok request: model=grok-3-mini, messages={len(messages)}, system={messages[0]['content'][:80] if messages else 'none'}...")

        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="grok-3-mini",
                messages=messages,
            ),
            timeout=60,
        )

        reply = (response.choices[0].message.content or "").strip()
        logger.info(f"GPT raw reply: {repr(reply)}, finish_reason={response.choices[0].finish_reason}")

        if not reply:
            return "эээ… я задумалась 😅"

        if dumb_mode:
            words = reply.split()
            reply = " ".join(words[:MAX_WORDS])

        reply = lowercase_first(reply)
        return reply

    except Exception as e:
        logger.error(f"Grok API error: {e}", exc_info=True)
        return "эээ… я зависла 😳"
