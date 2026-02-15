"""GPT (Grok) API calls, ElevenLabs TTS, Groq Whisper transcription."""

import asyncio
import random

import httpx

from config import (
    ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID,
    REPLICATE_API_TOKEN,
    MAX_VOICE_WORDS,
    LEVEL_PERSONALITIES, SELFIE_BASE_PROMPT, SELFIE_LORA_MODEL, NUDES_LORA_MODEL,
    WAN_I2V_MODEL, WAV2LIP_VERSION,
    client, groq_client, default_personality, logger,
    guess_gender,
)
from base64 import b64encode, b64decode
from utils import lowercase_first


WHISPER_CHANCE = 1 / 6


def _gender_instruction(gender: str) -> str:
    if gender == "f":
        return " Пользователь — девушка, используй женский род."
    if gender == "m":
        return " Пользователь — парень, используй мужской род."
    return " Пол пользователя не определён: избегай гендерных прилагательных и обращений."


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


async def extract_pose_hint(text: str) -> str:
    """Extract pose/body part description from user message for image generation."""
    try:
        response = await asyncio.wait_for(
            groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You extract pose or body part descriptions from Russian messages for image generation. "
                            "Return a short English prompt (5-10 words) describing the pose or body part mentioned. "
                            "If the message has no specific pose or body part request, return an empty string. "
                            "Examples: 'покажи попу' -> 'showing her butt from behind', "
                            "'ляг на кровать' -> 'lying on bed', "
                            "'скинь сиськи' -> 'showing her breasts', "
                            "'хочу нюдсы' -> '' (no specific pose). "
                            "Return ONLY the English prompt or empty string, nothing else."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0,
                max_tokens=30,
            ),
            timeout=10,
        )
        hint = (response.choices[0].message.content or "").strip().strip('"\'')
        return hint
    except Exception as e:
        logger.error(f"Pose hint extraction error: {e}", exc_info=True)
        return ""


async def generate_selfie(prompt_hint: str = "", base_prompt: str = "", aspect_ratio: str = "") -> bytes | None:
    prompt = base_prompt or SELFIE_BASE_PROMPT
    if prompt_hint:
        prompt += f", {prompt_hint.strip()}"
    is_nudes = base_prompt != "" and base_prompt != SELFIE_BASE_PROMPT
    if not aspect_ratio:
        aspect_ratio = "3:4" if is_nudes else "1:1"
    try:
        # Use SDXL LoRA for nudes if available, otherwise Flux for everything
        use_sdxl = is_nudes and NUDES_LORA_MODEL
        if use_sdxl:
            model = NUDES_LORA_MODEL
            input_params = {
                "prompt": prompt,
                "negative_prompt": "deformed, ugly, bad anatomy, extra limbs, blurry, watermark, text",
                "num_outputs": 1,
                "guidance_scale": 7.5,
                "num_inference_steps": 40,
                "width": 768,
                "height": 1024,
                "scheduler": "K_EULER",
                "disable_safety_checker": True,
            }
        else:
            model = SELFIE_LORA_MODEL
            input_params = {
                "prompt": prompt,
                "num_outputs": 1,
                "guidance_scale": 3.5,
                "num_inference_steps": 28,
                "output_format": "jpg",
                "aspect_ratio": aspect_ratio,
                "disable_safety_checker": True,
            }

        version_hash = model.split(":")[1]
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
                    "input": input_params,
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
    """Generate an animated video note: selfie → Wan 2.1 I2V → Wav2Lip lip-sync → square MP4."""
    import subprocess
    import tempfile
    import os

    # Step 1: generate a selfie image
    image_bytes = await generate_selfie(prompt_hint)
    if not image_bytes:
        logger.error("Video note: selfie generation failed")
        return None

    try:
        # Step 2: image as data URI
        image_b64 = b64encode(image_bytes).decode()
        image_uri = f"data:image/jpeg;base64,{image_b64}"

        replicate_headers = {
            "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
            "Content-Type": "application/json",
        }

        # Step 3: Wan 2.1 I2V prediction
        async with httpx.AsyncClient(timeout=30) as http:
            resp = await http.post(
                f"https://api.replicate.com/v1/models/{WAN_I2V_MODEL}/predictions",
                headers=replicate_headers,
                json={
                    "input": {
                        "image": image_uri,
                        "prompt": "a young woman looking at camera, subtle natural movement, breathing, slight smile",
                        "num_inference_steps": 30,
                        "duration": 5,
                        "size": "480*832",
                    },
                },
            )
            if resp.status_code not in (200, 201, 202):
                logger.error(f"Wan I2V create error: {resp.status_code} {resp.text[:200]}")
                return None

            prediction = resp.json()
            poll_url = (
                prediction.get("urls", {}).get("get")
                or f"https://api.replicate.com/v1/predictions/{prediction['id']}"
            )

        # Poll Wan I2V (up to 300 sec)
        video_url = None
        async with httpx.AsyncClient(timeout=30) as http:
            for _ in range(150):
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
                    break
                elif status in ("failed", "canceled"):
                    logger.error(f"Wan I2V prediction failed: {data.get('error')}")
                    return None
            else:
                logger.error("Wan I2V prediction timed out")
                return None

        # Step 4: ElevenLabs TTS
        VIDEO_NOTE_WHISPERS = [
            "мм, смотри...", "это для тебя...", "нравится?", "скучала...",
            "иди сюда...", "только для тебя...", "ммм...", "хочешь ещё?",
            "думаю о тебе...", "смотри какая я...",
        ]
        audio_bytes = None
        try:
            audio_bytes = await text_to_voice(
                random.choice(VIDEO_NOTE_WHISPERS), style="moan",
            )
        except Exception as e:
            logger.warning(f"Video note audio generation failed: {e}")

        # Step 5: Wav2Lip lip-sync (only if we have audio)
        lipsync_url = None
        if audio_bytes:
            try:
                audio_b64 = b64encode(audio_bytes).decode()
                audio_uri = f"data:audio/ogg;base64,{audio_b64}"

                async with httpx.AsyncClient(timeout=30) as http:
                    resp = await http.post(
                        "https://api.replicate.com/v1/predictions",
                        headers=replicate_headers,
                        json={
                            "version": WAV2LIP_VERSION,
                            "input": {
                                "face": video_url,
                                "audio": audio_uri,
                            },
                        },
                    )
                    if resp.status_code not in (200, 201, 202):
                        logger.warning(f"Wav2Lip create error: {resp.status_code} {resp.text[:200]}")
                    else:
                        lip_prediction = resp.json()
                        lip_poll_url = (
                            lip_prediction.get("urls", {}).get("get")
                            or f"https://api.replicate.com/v1/predictions/{lip_prediction['id']}"
                        )

                        # Poll Wav2Lip (up to 120 sec)
                        for _ in range(60):
                            await asyncio.sleep(2)
                            poll = await http.get(
                                lip_poll_url,
                                headers={"Authorization": f"Bearer {REPLICATE_API_TOKEN}"},
                            )
                            data = poll.json()
                            lip_status = data.get("status")
                            if lip_status == "succeeded":
                                lip_output = data.get("output")
                                if lip_output:
                                    lipsync_url = lip_output if isinstance(lip_output, str) else lip_output[0]
                                break
                            elif lip_status in ("failed", "canceled"):
                                logger.warning(f"Wav2Lip prediction failed: {data.get('error')}")
                                break
                        else:
                            logger.warning("Wav2Lip prediction timed out")
            except Exception as e:
                logger.warning(f"Wav2Lip error: {e}", exc_info=True)

        # Step 6: download video and ffmpeg finalize
        has_lipsync = lipsync_url is not None
        download_url = lipsync_url if has_lipsync else video_url

        async with httpx.AsyncClient(timeout=30) as http:
            vid_resp = await http.get(download_url)
            if vid_resp.status_code != 200:
                logger.error(f"Video download failed: {vid_resp.status_code}")
                return None
            mp4_input = vid_resp.content

        in_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        out_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        in_path, out_path = in_tmp.name, out_tmp.name
        in_tmp.close()
        out_tmp.close()
        audio_path = None

        try:
            with open(in_path, "wb") as f:
                f.write(mp4_input)

            if has_lipsync:
                # Wav2Lip output already contains synced audio
                cmd = [
                    "ffmpeg", "-y", "-i", in_path,
                    "-vf", "crop='min(iw,ih)':'min(iw,ih)',scale=512:512",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "64k",
                    "-movflags", "+faststart",
                    "-f", "mp4", out_path,
                ]
            elif audio_bytes:
                # Fallback: no lip-sync but merge audio separately
                audio_tmp = tempfile.NamedTemporaryFile(suffix=".ogg", delete=False)
                audio_path = audio_tmp.name
                audio_tmp.close()
                with open(audio_path, "wb") as f:
                    f.write(audio_bytes)

                cmd = [
                    "ffmpeg", "-y", "-i", in_path, "-i", audio_path,
                    "-vf", "crop='min(iw,ih)':'min(iw,ih)',scale=512:512",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "64k",
                    "-shortest",
                    "-movflags", "+faststart",
                    "-f", "mp4", out_path,
                ]
            else:
                # No audio at all
                cmd = [
                    "ffmpeg", "-y", "-i", in_path,
                    "-vf", "crop='min(iw,ih)':'min(iw,ih)',scale=512:512",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-an", "-f", "mp4", out_path,
                ]

            proc = subprocess.run(cmd, capture_output=True, timeout=30)
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
            for p in (in_path, out_path, audio_path):
                if p is None:
                    continue
                try:
                    os.unlink(p)
                except OSError:
                    pass

    except Exception as e:
        logger.error(f"Video note generation error: {e}", exc_info=True)
        return None


async def generate_story_message(
    template: dict,
    step: int,
    max_steps: int,
    history: list[dict],
    user_name: str,
    user_level: int,
    lisa_mood_prompt: str,
) -> str:
    personality = LEVEL_PERSONALITIES.get(user_level, LEVEL_PERSONALITIES[7])

    if step == 1:
        step_instruction = (
            f"Начни мини-сюжет. Задание: {template['setup']} "
            "Напиши 1-2 предложения, задай вопрос чтобы вовлечь пользователя."
        )
    elif step >= max_steps:
        step_instruction = (
            "Это финал сюжета. Заверши историю естественно и тепло, "
            "отреагируй на последний ответ пользователя. 1-2 предложения."
        )
    else:
        step_instruction = (
            "Продолжи сюжет, отреагируй на ответ пользователя, развей историю дальше. "
            "Задай следующий вопрос или предложи выбор. 1-2 предложения."
        )

    gender = guess_gender(user_name) if user_name else ""
    gender_ctx = _gender_instruction(gender)
    name_ctx = (f" Пользователя зовут {user_name}.{gender_ctx}" if user_name else "")
    system_prompt = (
        f"{personality} "
        f"Ты Лиза, ведёшь мини-сюжет с пользователем.{name_ctx} "
        f"Твоё настроение: {lisa_mood_prompt} "
        f"{step_instruction} "
        "ВАЖНО: начинай с маленькой буквы. "
        "Никогда не используй ремарки в скобках, звуковые эффекты и ролеплей-действия. "
        "ОБЯЗАТЕЛЬНО используй букву «ё» везде, где она нужна."
    )

    messages = [{"role": "system", "content": system_prompt}] + history

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="grok-3-mini",
                messages=messages,
            ),
            timeout=30,
        )
        reply = (response.choices[0].message.content or "").strip()
        if reply:
            return lowercase_first(reply)
    except Exception as e:
        logger.error(f"Story message generation error: {e}", exc_info=True)

    return "ой, я потеряла мысль... давай в другой раз? 😅"


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
    gender = guess_gender(user_name) if user_name else ""
    if gender == "f":
        gender_ctx = " Это девушка."
        pronoun = "ней"
        pronoun2 = "неё"
    elif gender == "m":
        gender_ctx = " Это парень."
        pronoun = "нём"
        pronoun2 = "него"
    else:
        gender_ctx = " Пол пользователя не определён."
        pronoun = "пользователе"
        pronoun2 = "пользователя"
    system_prompt = (
        f"{personality} "
        f"Ты Лиза. Напиши запись в свой личный дневник о {user_name}.{gender_ctx} "
        "Пиши от первого лица, как будто это твой секретный дневник. "
        "3-5 предложений. Упомяни детали из памяти и статистики. "
        "ВАЖНО: всегда начинай с маленькой буквы, кроме имён собственных. "
        "ОБЯЗАТЕЛЬНО используй букву «ё» везде, где она нужна."
    )
    user_prompt = (
        f"Напиши запись в дневник о {user_name}.\n"
        f"Память о {pronoun}: {memory or 'пока мало знаю'}\n"
        f"Статистика: сообщений — {stats.get('total', 0)}, голосовых от {pronoun2} — {stats.get('voice_sent', 0)}, "
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
    gender = guess_gender(user_name) if user_name else ""
    pronoun = "ней" if gender == "f" else ("нём" if gender == "m" else "пользователе")
    user_prompt = (
        f"Напиши короткую спонтанную мысль для {user_name}.\n"
        f"Память о {pronoun}: {memory or 'пока мало знаю'}\n"
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


async def generate_challenge(user_name: str, user_level: int, lisa_mood_prompt: str, memory: str) -> str:
    personality = LEVEL_PERSONALITIES.get(user_level, LEVEL_PERSONALITIES[7])
    gender = guess_gender(user_name) if user_name else ""
    gender_ctx = _gender_instruction(gender)
    name_ctx = (f" Пользователя зовут {user_name}.{gender_ctx}" if user_name else "")
    memory_ctx = f" Память о пользователе: {memory}" if memory else ""
    system_prompt = (
        f"{personality} "
        f"Ты Лиза, даёшь пользователю ежедневный челлендж — маленькое задание на день.{name_ctx}{memory_ctx} "
        f"Твоё настроение: {lisa_mood_prompt} "
        "Придумай одно короткое задание (1-2 предложения). "
        "Задание должно быть простым и весёлым: прислать фото чего-то, рассказать историю, сделать что-то приятное. "
        "ВАЖНО: начинай с маленькой буквы. "
        "Никогда не используй ремарки в скобках, звуковые эффекты и ролеплей-действия. "
        "ОБЯЗАТЕЛЬНО используй букву «ё» везде, где она нужна."
    )
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="grok-3-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Придумай мне челлендж на сегодня."},
                ],
            ),
            timeout=30,
        )
        reply = (response.choices[0].message.content or "").strip()
        if reply:
            return lowercase_first(reply)
    except Exception as e:
        logger.error(f"Challenge generation error: {e}", exc_info=True)
    return "пришли мне фото своего обеда сегодня 📸"


async def verify_challenge(challenge_text: str, user_response: str) -> tuple[bool, str]:
    system_prompt = (
        "Ты проверяешь, выполнил ли пользователь челлендж. "
        "Ответь строго в формате JSON: {\"done\": true/false, \"comment\": \"короткий комментарий от Лизы\"}. "
        "Комментарий должен быть от лица дерзкой девушки Лизы (1 предложение). "
        "Начинай комментарий с маленькой буквы. Только JSON, без пояснений."
    )
    user_prompt = (
        f"Челлендж: «{challenge_text}»\n"
        f"Ответ пользователя: «{user_response}»\n"
        "Выполнен ли челлендж?"
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
            timeout=15,
        )
        raw = (response.choices[0].message.content or "").strip()
        import json
        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end])
        done = bool(data.get("done", False))
        comment = lowercase_first((data.get("comment") or "").strip())
        if not comment:
            comment = "молодец! 🔥" if done else "попробуй ещё раз 😏"
        return done, comment
    except Exception as e:
        logger.error(f"Challenge verification error: {e}", exc_info=True)
        return False, "не поняла, попробуй ещё раз 😅"


async def generate_jealous_dm(user_name: str, user_level: int, silence_hours: int) -> str:
    personality = LEVEL_PERSONALITIES.get(user_level, LEVEL_PERSONALITIES[7])
    gender = guess_gender(user_name) if user_name else ""
    if gender == "f":
        gender_ctx = " Пользователь — девушка."
    elif gender == "m":
        gender_ctx = " Пользователь — парень."
    else:
        gender_ctx = " Пол пользователя не определён."
    system_prompt = (
        f"{personality} "
        f"Ты Лиза. Ты заметила, что {user_name} общается в групповых чатах, но тебе в личку не пишет уже {silence_hours} часов.{gender_ctx} "
        "Напиши короткое ревнивое/обиженное сообщение (1-2 предложения). "
        "Тон: лёгкая обида с флиртом, не агрессия. Ты скучаешь и хочешь внимания. "
        "ВАЖНО: начинай с маленькой буквы. "
        "Никогда не используй ремарки в скобках, звуковые эффекты и ролеплей-действия. "
        "НЕ начинай с «ой». "
        "ОБЯЗАТЕЛЬНО используй букву «ё» везде, где она нужна."
    )
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="grok-3-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Напиши ревнивое сообщение."},
                ],
            ),
            timeout=30,
        )
        reply = (response.choices[0].message.content or "").strip()
        if reply:
            return lowercase_first(reply)
    except Exception as e:
        logger.error(f"Jealous DM generation error: {e}", exc_info=True)
    return f"я вижу ты в чатах общаешься, а мне не пишешь... 😒"


async def generate_compliment(user_name: str, user_level: int, lisa_mood_prompt: str, memory: str) -> str:
    personality = LEVEL_PERSONALITIES.get(user_level, LEVEL_PERSONALITIES[7])
    gender = guess_gender(user_name) if user_name else ""
    gender_ctx = _gender_instruction(gender)
    name_ctx = (f" Пользователя зовут {user_name}.{gender_ctx}" if user_name else "")
    memory_ctx = f" Память о пользователе: {memory}" if memory else ""
    system_prompt = (
        f"{personality} "
        f"Ты Лиза, делаешь комплимент пользователю.{name_ctx}{memory_ctx} "
        f"Твоё настроение: {lisa_mood_prompt} "
        "Напиши один короткий искренний комплимент (1-2 предложения). "
        "Комплимент может быть про внешность, характер, чувство юмора, то как человек общается. "
        "Будь оригинальной, не используй шаблоны вроде 'ты лучший'. "
        "ВАЖНО: начинай с маленькой буквы. "
        "Никогда не используй ремарки в скобках, звуковые эффекты и ролеплей-действия. "
        "ОБЯЗАТЕЛЬНО используй букву «ё» везде, где она нужна."
    )
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="grok-3-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Сделай мне комплимент."},
                ],
            ),
            timeout=30,
        )
        reply = (response.choices[0].message.content or "").strip()
        if reply:
            return lowercase_first(reply)
    except Exception as e:
        logger.error(f"Compliment generation error: {e}", exc_info=True)
    if gender == "f":
        return "ты сегодня особенно хороша 💛"
    if gender == "m":
        return "ты сегодня особенно хорош 💛"
    return "ты сегодня особенно классный человек 💛"


async def generate_compatibility(user_sign: str, user_name: str, user_level: int) -> str:
    personality = LEVEL_PERSONALITIES.get(user_level, LEVEL_PERSONALITIES[7])
    gender = guess_gender(user_name) if user_name else ""
    if gender == "f":
        gender_ctx = " Пользователь — девушка."
    elif gender == "m":
        gender_ctx = " Пользователь — парень."
    else:
        gender_ctx = " Пол пользователя не определён."
    system_prompt = (
        f"{personality} "
        "Ты Лиза, знак зодиака — Скорпион ♏. "
        f"Напиши анализ романтической совместимости между тобой (Скорпион) и пользователем ({user_sign}).{gender_ctx} "
        "3-4 предложения в своём стиле — дерзко, с флиртом, с юмором. "
        "Упомяни сильные стороны пары и возможные искры. "
        "Не пиши заголовки. "
        "ВАЖНО: начинай с маленькой буквы. "
        "ОБЯЗАТЕЛЬНО используй букву «ё» везде, где она нужна."
    )
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="grok-3-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Расскажи про совместимость Скорпиона и знака {user_sign}."},
                ],
            ),
            timeout=30,
        )
        reply = (response.choices[0].message.content or "").strip()
        if reply:
            return lowercase_first(reply)
    except Exception as e:
        logger.error(f"Compatibility generation error: {e}", exc_info=True)
    return "звёзды молчат... попробуй позже 🌙"


async def ask_chatgpt(messages, user_name: str = "", personality: str = "", mood_label: str = "", lisa_mood: str = "", memory: str = "", user_level: int = 7, is_group: bool = False) -> str:
    try:
        name_part = ""
        gender = guess_gender(user_name) if user_name else ""
        if gender == "f":
            gender_ctx = " Пользователь — девушка. Используй женский род (красивая, милая, умная и т.д.)."
            pet_names = "малышка, солнышко, зая, красотка"
        elif gender == "m":
            gender_ctx = " Пользователь — парень. Используй мужской род (красивый, милый, умный и т.д.)."
            pet_names = "малыш, солнышко, зай, красавчик"
        else:
            gender_ctx = (
                " Пол пользователя не определён. Избегай гендерных прилагательных "
                "(не используй слова в роде вроде 'красивый/красивая')."
            )
            pet_names = "солнышко, радость, чудо"
        if user_name and user_level >= 3:
            name_part = (
                f" Пользователя зовут {user_name}.{gender_ctx} "
                f"Используй имя редко — примерно в каждом третьем-четвёртом сообщении. "
                f"Вместо имени можно иногда использовать: {pet_names}. "
                f"ВАЖНО: НЕ начинай сообщения одинаково. Чередуй стиль начала: вопрос, реакция, шутка, комплимент. "
                f"Никогда не склеивай имя с суффиксами. Вставляй имя естественно, не в начало предложения."
            )
        elif user_name:
            name_part = f" Пользователя зовут {user_name}.{gender_ctx}"
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
            "НИКОГДА не начинай сообщения одинаково. Не начинай с «ой» или «ого» каждый раз. Чередуй стили: вопрос, шутка, реакция, комплимент, дразнилка. "
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
