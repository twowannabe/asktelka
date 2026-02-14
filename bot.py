#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Companion Telegram bot ("милая девушка") — entry point.
All logic is split into modules: config, db, gpt, handlers, games, checkin, utils.
"""

from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

from config import (
    TELEGRAM_TOKEN, CHECK_LONELY_INTERVAL_SEC, LISA_MOOD_UPDATE_INTERVAL_SEC,
    RITUAL_CHECK_INTERVAL_SEC, THOUGHT_CHECK_INTERVAL_SEC, logger,
)
from db import init_db
from handlers import (
    start_cmd, help_cmd, stats_cmd, level_cmd, achievements_cmd,
    set_personality_cmd, dontwritefirst_cmd, writefirst_cmd,
    mood_cmd, clear_mood_cmd, mood_lisa_cmd, disable_cmd, reset_cmd,
    selfie_cmd, nudes_gen_cmd, circle_cmd, horoscope_cmd, diary_cmd,
    handle_message, handle_voice, handle_media, error_handler,
)
from games import truth_cmd, guess_cmd, riddle_cmd, quiz_cmd, quiz_callback
from checkin import check_lonely_users, update_lisa_mood, send_ritual, send_lisa_thoughts


def main():
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).read_timeout(60).build()

    init_db()

    application.add_handler(CommandHandler("start", start_cmd))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("disable", disable_cmd))
    application.add_handler(CommandHandler("reset", reset_cmd))
    application.add_handler(CommandHandler("set_personality", set_personality_cmd))

    application.add_handler(CommandHandler("dontwritefirst", dontwritefirst_cmd))
    application.add_handler(CommandHandler("writefirst", writefirst_cmd))
    application.add_handler(CommandHandler("mood", mood_cmd))
    application.add_handler(CommandHandler("clear_mood", clear_mood_cmd))
    application.add_handler(CommandHandler("stats", stats_cmd))
    application.add_handler(CommandHandler("level", level_cmd))
    application.add_handler(CommandHandler("achievements", achievements_cmd))
    application.add_handler(CommandHandler("selfie", selfie_cmd))
    application.add_handler(CommandHandler("circle", circle_cmd))
    application.add_handler(CommandHandler("nudes", nudes_gen_cmd))
    application.add_handler(CommandHandler("horoscope", horoscope_cmd))
    application.add_handler(CommandHandler("mood_lisa", mood_lisa_cmd))
    application.add_handler(CommandHandler("diary", diary_cmd))

    # Mini-games
    application.add_handler(CommandHandler("truth", truth_cmd))
    application.add_handler(CommandHandler("guess", guess_cmd))
    application.add_handler(CommandHandler("riddle", riddle_cmd))
    application.add_handler(CommandHandler("quiz", quiz_cmd))
    application.add_handler(CallbackQueryHandler(quiz_callback, pattern=r"^quiz_"))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    application.add_handler(MessageHandler(filters.PHOTO | filters.Sticker.ALL | filters.VIDEO | filters.ANIMATION, handle_media))
    application.add_error_handler(error_handler)

    # Schedule periodic lonely-user checks
    job_queue = application.job_queue
    job_queue.run_repeating(
        check_lonely_users,
        interval=CHECK_LONELY_INTERVAL_SEC,
        first=60,
    )
    job_queue.run_repeating(
        update_lisa_mood,
        interval=LISA_MOOD_UPDATE_INTERVAL_SEC,
        first=10,
    )
    job_queue.run_repeating(
        send_ritual,
        interval=RITUAL_CHECK_INTERVAL_SEC,
        first=30,
    )
    job_queue.run_repeating(
        send_lisa_thoughts,
        interval=THOUGHT_CHECK_INTERVAL_SEC,
        first=120,
    )

    logger.info("Bot started")
    application.run_polling()


if __name__ == "__main__":
    main()
