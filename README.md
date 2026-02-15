# Лиза — Telegram-бот-компаньон

Личный Telegram-бот с характером: общение через LLM, голосовые, мини-игры, XP/уровни, webapp-профиль и периодические инициативные сообщения.

## Стек

- Python 3.10+
- `python-telegram-bot` (polling + job queue)
- PostgreSQL (`psycopg2`)
- xAI Grok (чат, генерация игровых и контентных ответов)
- Groq Whisper (распознавание голосовых)
- ElevenLabs (TTS)
- Replicate (генерация изображений/видео для медиа-команд)

## Возможности

- Ответы в личке и группах (триггеры: упоминание, reply, вероятностные реакции).
- Обработка голосовых: транскрипция + генерация ответа.
- Голосовые ответы командой `/voice`.
- Система прогресса: XP, уровни, стрик, ачивки, топ пользователей.
- Мини-игры: truth/guess/riddle/quiz + интерактивный story-режим.
- WebApp-профиль (`/profile`) с настройками пользователя.
- Память: история диалогов, краткое саммари пользователя, настроение.
- Периодические задачи: check-in, обновление настроения Лизы, ритуалы/мысли/челленджи/комплименты.

## Команды

### Основное

- `/start` — приветствие.
- `/help` — список команд.
- `/stats` — статистика общения.
- `/level` — уровень и прогресс XP.
- `/achievements` — ачивки пользователя.
- `/profile` — webapp-профиль (или текстовый fallback).
- `/top` — рейтинг пользователей.
- `/diary` — короткая сводка/дневник отношений.
- `/mood_lisa` — текущее настроение Лизы.

### Медиа и AI

- `/selfie [подсказка]` — отправить селфи.
- `/nudes [описание]` — генерация фото.
- `/circle [описание]` — видео-кружочек (генерация или fallback из папки).
- `/voice [--стиль] текст` — голосовое сообщение.
- `/horoscope [знак]` — гороскоп.
- `/compatibility` — совместимость с Лизой.
- `/story` — интерактивный мини-сюжет.

### Игры

- `/challenge` — челлендж дня.
- `/truth` — правда или действие.
- `/guess` — угадай число.
- `/riddle` — загадка.
- `/quiz` — викторина с кнопками.

### Настройки

- `/set_personality <текст>` — пользовательский стиль.
- `/dontwritefirst` — не писать первой.
- `/writefirst` — снова писать первой.
- `/mood` — показать запомненное настроение.
- `/clear_mood` — очистить настроение.
- `/reset` — сбросить контекст диалога.
- `/disable` — выключить бота в текущем чате.

## Быстрый старт

### 1) Зависимости

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2) PostgreSQL

Нужна доступная база; таблицы и миграции создаются автоматически при старте (`init_db()`).

### 3) `.env`

Создай `.env` в корне проекта:

```env
TELEGRAM_TOKEN=123456:AAF...
XAI_API_KEY=xai-...
GROQ_API_KEY=gsk_...
ELEVENLABS_API_KEY=...
ELEVENLABS_VOICE_ID=...

DB_HOST=127.0.0.1
DB_PORT=5432
DB_NAME=companion_bot
DB_USER=postgres
DB_PASSWORD=yourpassword

# optional
REPLICATE_API_TOKEN=
NEWS_RSS_URL=
WEBAPP_URL=
NUDES_LORA_MODEL=
```

### 4) Запуск

```bash
python3 bot.py
```

## Деплой

В репозитории есть GitHub Actions workflow: `.github/workflows/deploy.yml`.

Текущее поведение:
- Триггер: push в `main`.
- SSH на сервер.
- `git pull origin main`.
- `pip install -r requirements.txt`.
- `systemctl restart asktelka`.

Для systemd используется юнит `asktelka.service`.

## Структура проекта

- `bot.py` — точка входа, регистрация хендлеров, планировщик задач.
- `handlers.py` — команды и обработчики сообщений/медиа/webapp.
- `gpt.py` — интеграции LLM/TTS/STT/генерация контента.
- `db.py` — пул соединений, миграции, CRUD.
- `checkin.py` — фоновые инициативные сообщения.
- `games.py` — логика мини-игр.
- `config.py` — все константы, клиенты API, runtime state.
- `webapp/index.html` — мини-приложение профиля.

## Примечания

- Для корректной перекодировки голосовых желательно наличие `ffmpeg` в системе.
- Проект ориентирован на long-running процесс (polling), а не webhook.
