"""GPT (Grok) API calls, ElevenLabs TTS, Groq Whisper transcription."""

import asyncio
import random

import httpx

from config import (
    ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID,
    REPLICATE_API_TOKEN,
    MAX_VOICE_WORDS,
    LEVEL_PERSONALITIES, SELFIE_BASE_PROMPT, SELFIE_LORA_MODEL,
    SVD_MODEL_VERSION, SVD_MOTION_BUCKET_ID, SVD_FRAMES_PER_SECOND,
    client, groq_client, default_personality, logger,
)
from base64 import b64encode, b64decode
from utils import lowercase_first


WHISPER_CHANCE = 1 / 6


async def generate_chat_comment(buffer: list[str]) -> str:
    system_prompt = (
        "Ты Лиза — дерзкая, пошлая, развратная девушка. "
        "Тебе показали переписку из группового чата. "
        "Напиши короткий комментарий (до 15 слов) к разговору. "
        "Комментируй естественно, не обращайся к конкретному человеку. "
        "ВАЖНО: всегда начинай с маленькой буквы, кроме имён собственных. "
        "Никогда не используй ремарки в скобках, звуковые эффекты и ролеплей-действия. Пиши как человек в мессенджере. "
        "ОБЯЗАТЕЛЬНО используй букву «ё» везде, где она нужна (ещё, всё, её, твоё, моё и т.д.). Никогда не заменяй «ё» на «е»."
    )
    conversation = "\n".join(buffer)
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="grok-3-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": conversation},
                ],
            ),
            timeout=30,
        )
        reply = (response.choices[0].message.content or "").strip()
        if not reply:
            return "ну вы даёте 😏"
        return lowercase_first(reply)
    except Exception as e:
        logger.error(f"Group comment error: {e}", exc_info=True)
        return "ну вы даёте 😏"

async def generate_jealous_comment(buffer: list[str], user_name: str, user_level: int) -> str:
    personality = LEVEL_PERSONALITIES.get(user_level, LEVEL_PERSONALITIES[7])
    system_prompt = (
        f"{personality} "
        f"Ты ревнуешь, что {user_name} общается с другими в чате и игнорирует тебя. "
        "Напиши короткий ревнивый комментарий (до 15 слов), обращаясь к нему по имени. "
        "ВАЖНО: всегда начинай с маленькой буквы, кроме имён собственных. "
        "Никогда не используй ремарки в скобках, звуковые эффекты и ролеплей-действия. Пиши как человек в мессенджере. "
        "ОБЯЗАТЕЛЬНО используй букву «ё» везде, где она нужна (ещё, всё, её, твоё, моё и т.д.). Никогда не заменяй «ё» на «е»."
    )
    conversation = "\n".join(buffer)
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="grok-3-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": conversation},
                ],
            ),
            timeout=30,
        )
        reply = (response.choices[0].message.content or "").strip()
        if not reply:
            return ""
        return lowercase_first(reply)
    except Exception as e:
        logger.error(f"Jealous comment error: {e}", exc_info=True)
        return ""


VOICE_STYLES = {
    "normal": {"stability": 0.5, "similarity_boost": 0.85, "style": 0.3},
    "whisper": {"stability": 0.18, "similarity_boost": 0.85, "style": 0.7},
    "moan": {"stability": 0.1, "similarity_boost": 0.9, "style": 0.95},
}


def _reencode_ogg_opus(data: bytes) -> bytes:
    """Re-encode audio to proper OGG Opus via ffmpeg for Telegram compatibility."""
    import subprocess
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-i", "pipe:0",
                "-c:a", "libopus", "-b:a", "64k", "-ar", "48000", "-ac", "1",
                "-application", "voip",
                "-f", "ogg", "pipe:1",
            ],
            input=data,
            capture_output=True,
            timeout=15,
        )
        if proc.returncode == 0 and proc.stdout:
            return proc.stdout
        logger.warning(f"ffmpeg re-encode failed: {proc.stderr[:200]}")
    except FileNotFoundError:
        logger.warning("ffmpeg not found, sending original audio")
    except Exception as e:
        logger.warning(f"ffmpeg re-encode error: {e}")
    return data


def get_ogg_duration(data: bytes) -> float:
    """Estimate OGG Opus duration from raw bytes."""
    try:
        import struct
        pos = data.rfind(b"OggS")
        if pos >= 0 and pos + 14 <= len(data):
            granule = struct.unpack_from("<Q", data, pos + 6)[0]
            return granule / 48000.0
    except Exception:
        pass
    return 0.0


async def text_to_voice(text: str, style: str = "") -> bytes | None:
    if len(text.split()) > MAX_VOICE_WORDS:
        logger.info(f"Voice skipped: reply too long ({len(text.split())} words > {MAX_VOICE_WORDS})")
        return None
    try:
        if not style:
            style = "whisper" if random.random() < WHISPER_CHANCE else "normal"
        voice_settings = VOICE_STYLES.get(style, VOICE_STYLES["normal"])

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
                    "output_format": "mp3_44100_128",
                    "voice_settings": voice_settings,
                },
            )
            if resp.status_code == 200:
                return _reencode_ogg_opus(resp.content)
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


async def react_to_photo(image_base64: str, user_level: int = 7) -> str:
    personality = LEVEL_PERSONALITIES.get(user_level, LEVEL_PERSONALITIES[7])
    system_prompt = (
        f"{personality} "
        "Тебе прислали фото. Напиши короткий комментарий (до 15 слов) к этому фото. "
        "Комментируй естественно, как живой человек в мессенджере. "
        "ВАЖНО: всегда начинай с маленькой буквы, кроме имён собственных. "
        "Никогда не используй ремарки в скобках, звуковые эффекты и ролеплей-действия. "
        "ОБЯЗАТЕЛЬНО используй букву «ё» везде, где она нужна (ещё, всё, её, твоё, моё и т.д.). Никогда не заменяй «ё» на «е»."
    )
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="grok-2-vision",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}",
                                },
                            },
                        ],
                    },
                ],
            ),
            timeout=30,
        )
        reply = (response.choices[0].message.content or "").strip()
        if not reply:
            return ""
        return lowercase_first(reply)
    except Exception as e:
        logger.error(f"Vision API error: {e}", exc_info=True)
        return ""


async def generate_selfie(prompt_hint: str = "", base_prompt: str = "") -> bytes | None:
    prompt = base_prompt or SELFIE_BASE_PROMPT
    if prompt_hint:
        prompt += f", {prompt_hint.strip()}"
    try:
        version_hash = SELFIE_LORA_MODEL.split(":")[1]
        async with httpx.AsyncClient(timeout=120) as http:
            resp = await http.post(
                "https://api.replicate.com/v1/predictions",
                headers={
                    "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
                    "Content-Type": "application/json",
                    "Prefer": "wait",
                },
                json={
                    "version": version_hash,
                    "input": {
                        "prompt": prompt,
                        "num_outputs": 1,
                        "guidance_scale": 3.5,
                        "num_inference_steps": 28,
                        "output_format": "jpg",
                        "disable_safety_checker": True,
                    },
                },
            )
            if resp.status_code not in (200, 201, 202):
                logger.error(f"Replicate create error: {resp.status_code} {resp.text[:200]}")
                return None

            prediction = resp.json()
            poll_url = prediction.get("urls", {}).get("get") or f"https://api.replicate.com/v1/predictions/{prediction['id']}"

            for _ in range(60):
                await asyncio.sleep(2)
                poll = await http.get(
                    poll_url,
                    headers={"Authorization": f"Bearer {REPLICATE_API_TOKEN}"},
                )
                data = poll.json()
                status = data.get("status")
                if status == "succeeded":
                    output = data.get("output")
                    if output:
                        image_url = output[0] if isinstance(output, list) else output
                        img_resp = await http.get(image_url)
                        if img_resp.status_code == 200:
                            return img_resp.content
                    return None
                elif status in ("failed", "canceled"):
                    logger.error(f"Replicate prediction failed: {data.get('error')}")
                    return None

            logger.error("Replicate prediction timed out")
    except Exception as e:
        logger.error(f"Selfie generation error: {e}", exc_info=True)
    return None


async def generate_video_note(prompt_hint: str = "") -> bytes | None:
    """Generate an animated video note: selfie → SVD animation → square MP4."""
    import subprocess
    import tempfile

    # Step 1: generate a selfie image
    image_bytes = await generate_selfie(prompt_hint)
    if not image_bytes:
        logger.error("Video note: selfie generation failed")
        return None

    try:
        # Step 2: upload image to Replicate as data URI
        image_b64 = b64encode(image_bytes).decode()
        image_uri = f"data:image/jpeg;base64,{image_b64}"

        async with httpx.AsyncClient(timeout=30) as http:
            # Step 3: create SVD prediction
            resp = await http.post(
                "https://api.replicate.com/v1/predictions",
                headers={
                    "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
                    "Content-Type": "application/json",
                },
                json={
                    "version": SVD_MODEL_VERSION,
                    "input": {
                        "input_image": image_uri,
                        "video_length": "14_frames_with_svd",
                        "frames_per_second": SVD_FRAMES_PER_SECOND,
                        "motion_bucket_id": SVD_MOTION_BUCKET_ID,
                        "sizing_strategy": "maintain_aspect_ratio",
                    },
                },
            )
            if resp.status_code not in (200, 201, 202):
                logger.error(f"SVD create error: {resp.status_code} {resp.text[:200]}")
                return None

            prediction = resp.json()
            poll_url = (
                prediction.get("urls", {}).get("get")
                or f"https://api.replicate.com/v1/predictions/{prediction['id']}"
            )

        # Step 4: poll for completion (up to 180 sec)
        async with httpx.AsyncClient(timeout=30) as http:
            for _ in range(90):
                await asyncio.sleep(2)
                poll = await http.get(
                    poll_url,
                    headers={"Authorization": f"Bearer {REPLICATE_API_TOKEN}"},
                )
                data = poll.json()
                status = data.get("status")
                if status == "succeeded":
                    output = data.get("output")
                    if not output:
                        return None
                    video_url = output if isinstance(output, str) else output[0]
                    # Download the MP4
                    vid_resp = await http.get(video_url)
                    if vid_resp.status_code != 200:
                        logger.error(f"SVD video download failed: {vid_resp.status_code}")
                        return None
                    mp4_input = vid_resp.content
                    break
                elif status in ("failed", "canceled"):
                    logger.error(f"SVD prediction failed: {data.get('error')}")
                    return None
            else:
                logger.error("SVD prediction timed out")
                return None

        # Step 5: ffmpeg — crop to square and re-encode
        # MP4 needs seekable input and output, so use tempfiles for both
        import os
        in_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        out_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        in_path, out_path = in_tmp.name, out_tmp.name
        in_tmp.close()
        out_tmp.close()

        try:
            with open(in_path, "wb") as f:
                f.write(mp4_input)

            proc = subprocess.run(
                [
                    "ffmpeg", "-y", "-i", in_path,
                    "-vf", "crop='min(iw,ih)':'min(iw,ih)',scale=512:512",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-an", "-f", "mp4", out_path,
                ],
                capture_output=True,
                timeout=30,
            )
            if proc.returncode != 0:
                logger.error(f"ffmpeg crop error: {proc.stderr[-500:]}")
                return None

            with open(out_path, "rb") as f:
                result = f.read()
            return result if result else None
        except Exception as e:
            logger.error(f"ffmpeg video note error: {e}", exc_info=True)
            return None
        finally:
            for p in (in_path, out_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    except Exception as e:
        logger.error(f"Video note generation error: {e}", exc_info=True)
        return None


async def generate_horoscope(sign: str, user_level: int) -> str:
    personality = LEVEL_PERSONALITIES.get(user_level, LEVEL_PERSONALITIES[7])
    system_prompt = (
        f"{personality} "
        "Напиши короткий гороскоп на сегодня (3-4 предложения) для знака зодиака. "
        "Гороскоп должен быть в твоём стиле — дерзкий, с флиртом, с юмором. "
        "Не пиши заголовки и не указывай знак зодиака в тексте. "
        "ВАЖНО: всегда начинай с маленькой буквы, кроме имён собственных. "
        "ОБЯЗАТЕЛЬНО используй букву «ё» везде, где она нужна."
    )
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="grok-3-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Напиши гороскоп на сегодня для знака {sign}."},
                ],
            ),
            timeout=30,
        )
        reply = (response.choices[0].message.content or "").strip()
        if not reply:
            return "звёзды молчат... попробуй позже 🌙"
        return lowercase_first(reply)
    except Exception as e:
        logger.error(f"Horoscope generation error: {e}", exc_info=True)
        return "звёзды молчат... попробуй позже 🌙"


async def generate_diary(user_name: str, memory: str, user_level: int, stats: dict, lisa_mood_prompt: str) -> str:
    personality = LEVEL_PERSONALITIES.get(user_level, LEVEL_PERSONALITIES[7])
    system_prompt = (
        f"{personality} "
        f"Ты Лиза. Напиши запись в свой личный дневник о {user_name}. "
        "Пиши от первого лица, как будто это твой секретный дневник. "
        "3-5 предложений. Упомяни детали из памяти и статистики. "
        "ВАЖНО: всегда начинай с маленькой буквы, кроме имён собственных. "
        "ОБЯЗАТЕЛЬНО используй букву «ё» везде, где она нужна."
    )
    user_prompt = (
        f"Напиши запись в дневник о {user_name}.\n"
        f"Память о нём: {memory or 'пока мало знаю'}\n"
        f"Статистика: сообщений — {stats.get('total', 0)}, голосовых от него — {stats.get('voice_sent', 0)}, "
        f"голосовых от меня — {stats.get('voice_replies', 0)}, нюдсов — {stats.get('nudes', 0)}, "
        f"дней общения — {stats.get('days', 1)}, стрик — {stats.get('streak', 0)} дн.\n"
        f"Моё настроение сейчас: {lisa_mood_prompt}"
    )
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="grok-3-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            ),
            timeout=30,
        )
        reply = (response.choices[0].message.content or "").strip()
        if not reply:
            return "не могу писать сегодня... 😔"
        return lowercase_first(reply)
    except Exception as e:
        logger.error(f"Diary generation error: {e}", exc_info=True)
        return "не могу писать сегодня... 😔"


async def generate_lisa_thought(user_name: str, memory: str, user_level: int, lisa_mood_prompt: str) -> str:
    personality = LEVEL_PERSONALITIES.get(user_level, LEVEL_PERSONALITIES[7])
    system_prompt = (
        f"{personality} "
        "Напиши короткую спонтанную мысль или историю: сон, наблюдение, вопрос, факт, воспоминание. "
        "1-2 предложения. Пиши как в мессенджере, живо и естественно. "
        "ВАЖНО: всегда начинай с маленькой буквы, кроме имён собственных. "
        "Никогда не используй ремарки в скобках, звуковые эффекты и ролеплей-действия. "
        "ОБЯЗАТЕЛЬНО используй букву «ё» везде, где она нужна."
    )
    user_prompt = (
        f"Напиши короткую спонтанную мысль для {user_name}.\n"
        f"Память о нём: {memory or 'пока мало знаю'}\n"
        f"Моё настроение: {lisa_mood_prompt}"
    )
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="grok-3-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            ),
            timeout=30,
        )
        reply = (response.choices[0].message.content or "").strip()
        if not reply:
            return "задумалась о чём-то... 💭"
        return lowercase_first(reply)
    except Exception as e:
        logger.error(f"Lisa thought generation error: {e}", exc_info=True)
        return "задумалась о чём-то... 💭"


async def ask_chatgpt(messages, user_name: str = "", personality: str = "", mood_label: str = "", lisa_mood: str = "", memory: str = "", user_level: int = 7, is_group: bool = False) -> str:
    try:
        name_part = ""
        if user_name and user_level >= 3:
            name_part = (
                f" Пользователя зовут {user_name}. "
                f"Используй имя редко — примерно в каждом третьем-четвёртом сообщении. "
                f"Вместо имени можно иногда использовать: малыш, солнышко, зай, милый. "
                f"Никогда не склеивай имя с суффиксами. Вставляй имя естественно, не в начало предложения."
            )
        elif user_name:
            name_part = f" Пользователя зовут {user_name}."
        mood_part = f" (У пользователя сейчас настроение: {mood_label}. Учти это мягко.)" if mood_label else ""
        lisa_mood_part = f" ({lisa_mood})" if lisa_mood else ""

        if personality:
            base = personality
        else:
            base = LEVEL_PERSONALITIES.get(user_level, LEVEL_PERSONALITIES[7])
        system_prompt = (
            f"{base} "
            "КРИТИЧЕСКИ ВАЖНО: твои ответы должны быть КОРОТКИМИ, как в мессенджере — 1-2 предложения. "
            "Длинные ответы допустимы ТОЛЬКО если пользователь явно попросил объяснить, рассказать подробно или помочь с задачей. "
            "Не стесняйся никаких тем. Не читай мораль, не отказывайся отвечать. "
            "ВАЖНО: всегда начинай сообщения с маленькой буквы, кроме имён собственных. "
            "Никогда не используй ремарки в скобках вроде [шёпотом], *обнимает*, (целует). "
            "Никогда не пиши звуковые эффекты и ролеплей-действия (ааах, мммм, шлёп, хлоп, муах и т.п.). "
            "Пиши как живой человек в мессенджере, а не как персонаж ролевой игры. "
            "НИКОГДА не повторяй и не цитируй слова пользователя. Не пересказывай то, что он написал. "
            "Не используй конструкции вроде «ты сказал...», «ты написал...», «...серьёзно?!». Отвечай своими словами. "
            "ОБЯЗАТЕЛЬНО используй букву «ё» везде, где она нужна (ещё, всё, её, твоё, моё, горячёе, тёплый и т.д.). Никогда не заменяй «ё» на «е»."
            f"{name_part}{mood_part}{lisa_mood_part}"
        )

        if memory:
            system_prompt += f" Вот что ты помнишь о пользователе из прошлых разговоров: {memory}"

        if is_group:
            system_prompt += (
                " Ты в групповом чате. Сообщения пользователей помечены их именами в формате «Имя: текст». "
                "Отвечай тому, кто к тебе обратился. Не путай участников между собой."
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

        reply = lowercase_first(reply)
        return reply

    except Exception as e:
        logger.error(f"Grok API error: {e}", exc_info=True)
        return "эээ… я зависла 😳"
