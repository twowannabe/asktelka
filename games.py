"""Mini-games: truth, guess, riddle, quiz."""

import asyncio
import json
import random

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes

from config import (
    XP_PER_TRUTH, XP_PER_GUESS, XP_PER_RIDDLE, XP_PER_QUIZ, XP_PER_STORY, XP_PER_CHALLENGE,
    STORY_TEMPLATES, LISA_MOODS,
    active_games, get_casual_name, client, logger,
)
from db import add_xp, send_level_up, get_user_level_info, get_lisa_mood
from utils import lowercase_first


async def truth_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    choice = random.choice(["правда", "действие"])

    prompt = (
        "Ты Лиза — дерзкая, пошлая девушка. "
        f"Придумай одно {'пошлое/дерзкое задание' if choice == 'действие' else 'пошлый/дерзкий вопрос'} для игры «правда или действие». "
        "Коротко, 1-2 предложения. Начни с маленькой буквы."
    )

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="grok-3-mini",
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=30,
        )
        question = lowercase_first((response.choices[0].message.content or "").strip())
    except Exception as e:
        logger.error(f"Truth game GPT error: {e}")
        question = "расскажи свою самую стыдную историю 😏" if choice == "правда" else "отправь своё самое смешное фото 😈"

    active_games[user_id] = {"type": "truth", "waiting": True}
    emoji = "❓" if choice == "правда" else "🎬"
    await update.message.reply_text(f"{emoji} {choice}!\n\n{question}")


async def guess_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    number = random.randint(1, 100)
    active_games[user_id] = {"type": "guess", "number": number, "attempts": 0}
    await update.message.reply_text("я загадала число от 1 до 100 😏 попробуй угадать!")


async def riddle_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    prompt = (
        "Ты Лиза — дерзкая девушка. Придумай одну загадку (не слишком сложную). "
        "Ответь строго в формате JSON: {\"riddle\": \"текст загадки\", \"answer\": \"ответ\"}. "
        "Начинай текст загадки с маленькой буквы. Только JSON, без пояснений."
    )

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="grok-3-mini",
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=30,
        )
        raw = (response.choices[0].message.content or "").strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end])
        riddle_text = lowercase_first(data["riddle"])
        answer = data["answer"].strip().lower()
    except Exception as e:
        logger.error(f"Riddle GPT error: {e}")
        riddle_text = "что можно держать без рук? 😏"
        answer = "обещание"

    active_games[user_id] = {"type": "riddle", "answer": answer, "waiting": True}
    await update.message.reply_text(f"🧩 загадка от Лизы:\n\n{riddle_text}")


async def quiz_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    prompt = (
        "Ты Лиза. Придумай один интересный вопрос-викторину с 4 вариантами ответа. "
        "Ответь строго в JSON: {\"question\": \"текст вопроса\", \"options\": {\"A\": \"...\", \"B\": \"...\", \"C\": \"...\", \"D\": \"...\"}, \"correct\": \"A\"}. "
        "Начинай вопрос с маленькой буквы. Только JSON."
    )

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="grok-3-mini",
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=30,
        )
        raw = (response.choices[0].message.content or "").strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end])
        question = lowercase_first(data["question"])
        options = data["options"]
        correct = data["correct"].upper()
    except Exception as e:
        logger.error(f"Quiz GPT error: {e}")
        question = "какая планета самая большая в солнечной системе?"
        options = {"A": "Марс", "B": "Юпитер", "C": "Сатурн", "D": "Земля"}
        correct = "B"

    active_games[user_id] = {"type": "quiz", "correct": correct}

    keyboard = [
        [InlineKeyboardButton(f"{k}: {v}", callback_data=f"quiz_{user_id}_{k}")]
        for k, v in options.items()
    ]
    markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(f"📝 викторина от Лизы:\n\n{question}", reply_markup=markup)


async def quiz_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    data = query.data
    parts = data.split("_")
    if len(parts) != 3:
        return

    target_user_id = int(parts[1])
    chosen = parts[2].upper()

    if query.from_user.id != target_user_id:
        await query.answer("это не твоя викторина 😏", show_alert=True)
        return

    game = active_games.pop(target_user_id, None)
    if not game or game["type"] != "quiz":
        await query.edit_message_text("эта викторина уже закончилась 🤷‍♀️")
        return

    chat_id = query.message.chat_id
    if chosen == game["correct"]:
        _, new_level, leveled_up = add_xp(target_user_id, XP_PER_QUIZ)
        await query.edit_message_text(f"✅ правильно! +{XP_PER_QUIZ} XP 🎉")
        if leveled_up:
            await send_level_up(context.bot, chat_id, new_level)
    else:
        await query.edit_message_text(f"❌ неправильно! правильный ответ: {game['correct']} 😏")


async def handle_game_response(user_id: int, text: str, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    game = active_games.get(user_id)
    if not game:
        return False

    chat_id = update.effective_chat.id

    if game["type"] == "truth":
        active_games.pop(user_id, None)
        prompt = (
            "Ты Лиза — дерзкая, пошлая девушка. "
            f"Пользователь ответил на правду/действие: «{text}». "
            "Отреагируй коротко (1-2 предложения), с юмором и дерзостью. Начни с маленькой буквы."
        )
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model="grok-3-mini",
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=30,
            )
            reaction = lowercase_first((response.choices[0].message.content or "").strip())
        except Exception:
            reaction = "ого, ну ты даёшь 😏"

        _, new_level, leveled_up = add_xp(user_id, XP_PER_TRUTH)
        await update.message.reply_text(f"{reaction}\n\n+{XP_PER_TRUTH} XP ⭐")
        if leveled_up:
            await send_level_up(context.bot, chat_id, new_level)
        return True

    elif game["type"] == "guess":
        try:
            guess = int(text.strip())
        except ValueError:
            await update.message.reply_text("напиши число от 1 до 100 🙄")
            return True

        game["attempts"] += 1
        target = game["number"]

        if guess == target:
            active_games.pop(user_id, None)
            attempts = game["attempts"]
            bonus = " бонус за скорость! 🚀" if attempts < 5 else ""
            xp = XP_PER_GUESS + (2 if attempts < 5 else 0)
            _, new_level, leveled_up = add_xp(user_id, xp)
            await update.message.reply_text(
                f"🎉 угадал за {attempts} попыток!{bonus}\n\n+{xp} XP ⭐"
            )
            if leveled_up:
                await send_level_up(context.bot, chat_id, new_level)
        elif guess < target:
            comment = random.choice(["больше 😏", "бери выше, малыш", "холодно... больше!", "нее, больше 🔥"])
            await update.message.reply_text(comment)
        else:
            comment = random.choice(["меньше 😏", "поменьше, зай", "горячо... но меньше!", "нет, меньше 🔥"])
            await update.message.reply_text(comment)
        return True

    elif game["type"] == "riddle":
        active_games.pop(user_id, None)
        correct_answer = game["answer"]

        prompt = (
            f"Правильный ответ на загадку: «{correct_answer}». "
            f"Пользователь ответил: «{text}». "
            "Это правильный ответ или достаточно близкий? Ответь строго: YES или NO."
        )
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model="grok-3-mini",
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=15,
            )
            verdict = (response.choices[0].message.content or "").strip().upper()
            is_correct = "YES" in verdict
        except Exception:
            is_correct = text.strip().lower() in correct_answer or correct_answer in text.strip().lower()

        if is_correct:
            _, new_level, leveled_up = add_xp(user_id, XP_PER_RIDDLE)
            await update.message.reply_text(f"✅ правильно, умничка! +{XP_PER_RIDDLE} XP 🎉")
            if leveled_up:
                await send_level_up(context.bot, chat_id, new_level)
        else:
            await update.message.reply_text(f"❌ неа, правильный ответ: {correct_answer} 😏")
        return True

    elif game["type"] == "challenge":
        from gpt import verify_challenge

        challenge_text = game["challenge"]
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        done, comment = await verify_challenge(challenge_text, text)

        if done:
            active_games.pop(user_id, None)
            _, new_level, leveled_up = add_xp(user_id, XP_PER_CHALLENGE)
            await update.message.reply_text(f"{comment}\n\n+{XP_PER_CHALLENGE} XP ⭐")
            if leveled_up:
                await send_level_up(context.bot, chat_id, new_level)
        else:
            await update.message.reply_text(comment)
        return True

    elif game["type"] == "story":
        from gpt import generate_story_message

        template_key = game["template"]
        template = next((t for t in STORY_TEMPLATES if t["key"] == template_key), None)
        if not template:
            active_games.pop(user_id, None)
            return True

        game["history"].append({"role": "user", "content": text})
        game["step"] += 1

        user_name = get_casual_name(update.effective_user.first_name or "")
        level_info = get_user_level_info(user_id)
        user_level = level_info["level"]
        lisa_mood_key = get_lisa_mood()
        lisa_mood_data = LISA_MOODS.get(lisa_mood_key, LISA_MOODS["playful"])

        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        await asyncio.sleep(random.uniform(1, 3))

        reply = await generate_story_message(
            template=template,
            step=game["step"],
            max_steps=game["max_steps"],
            history=game["history"],
            user_name=user_name,
            user_level=user_level,
            lisa_mood_prompt=lisa_mood_data["prompt_mod"],
        )

        game["history"].append({"role": "assistant", "content": reply})

        if game["step"] >= game["max_steps"]:
            active_games.pop(user_id, None)
            _, new_level, leveled_up = add_xp(user_id, XP_PER_STORY)
            await update.message.reply_text(f"{reply}\n\n+{XP_PER_STORY} XP ⭐")
            if leveled_up:
                await send_level_up(context.bot, chat_id, new_level)
        else:
            await update.message.reply_text(reply)

        return True

    return False
