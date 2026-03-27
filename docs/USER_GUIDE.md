# User Guide / Руководство пользователя

## RU: Что это

Podcast Reels Forge — инструмент для автоматического поиска «вирусных» моментов в длинном видео/подкасте и нарезки вертикальных клипов для Reels/Shorts/TikTok.

Пайплайн (по умолчанию):
1. Транскрибация видео/аудио через `faster-whisper` (локально, CUDA/CPU)
2. Поиск моментов через LLM (по умолчанию — Ollama)
3. Нарезка видео через FFmpeg (опционально 9:16)

Важно:
- Анализ выполняется по 5 фиксированным моделям (см. `config.yaml`).
- Результаты раскладываются по папкам `output/<model>/`.
- Если одна модель «упала»/зависла — пайплайн продолжит работу с другими моделями.
- Перед транскрибацией Forge проверяет, есть ли рядом с видео одноимённый `mp3`; если его нет, он автоматически извлекает аудио в `video.mp3` с битрейтом 320k и продолжает работу как обычно.
- По умолчанию Forge вшивает субтитры через `pycaps`, используя шрифт `assets/fonts/bignoodletoooblique.ttf` (меняется в `config.yaml`).

---

## RU: Быстрый старт

### 1) Требования

- Python 3.10+
- FFmpeg в системе (`ffmpeg` и `ffprobe` в PATH)
- Ollama (если используете локальные модели)
- CUDA (опционально, для ускорения транскрибации)

### 2) Установка

```bash
python3 -m venv whisper-env
source whisper-env/bin/activate
pip install -r requirements.txt
playwright install chromium
```

### 3) Подготовка входных данных

Положите видео в папку `input/` (например `input/video.mp4`). Пайплайн возьмёт самый новый файл.

### 4) Запуск полного пайплайна

```bash
python3 start_forge.py
```

Полезные флаги:
- `--verbose` — подробные логи
- `--quiet` — только ошибки
- `--no-skip-existing` — не пропускать стадии даже если результаты уже есть

---

## RU: Где лежат результаты

Результаты для каждой модели:
- `output/<model>/moments.json` — найденные моменты
- `output/<model>/reels.md` — описание/заголовки/мета (если модель вернула)
- `output/<model>/reels/` — нарезанные ролики (если включён шаг нарезки)
- `output/<model>/reels/reel_XX.srt` — локальный SRT для конкретного ролика, который используется для вшитых сабов
- `output/<model>/reels/reel_XX.md` — готовое описание до 1000 символов + 5 хештегов для каждого ролика

Папки моделей фиксированы и короткие:
- `qwen3`, `deepseek`, `gemma3`, `gemma2`, `gemini3`

---

## RU: Перегенерация видео из готовых moments.json (без LLM)

Это самый быстрый способ «переупаковать» видео (правильный вертикальный формат/кодек/битрейт), не трогая анализ.

### Команда

```bash
python3 rerender_videos.py --help
python3 rerender_videos.py --model gemma3
```

По умолчанию:
- читает `output/<model>/moments.json`
- пишет клипы в `output/<model>/reels_rerendered/`
- при включённых `subtitles.enabled: true` также вшивает сабы через `pycaps`
- рядом с каждым клипом создаёт `reel_XX.md` с описанием и 5 хештегами
- если файл уже существует, создаёт новый с суффиксом `_2`, `_3`, ... (ничего не затирает)

Чтобы перезаписать:

```bash
python3 rerender_videos.py --model gemma3 --replace
```

### Параметры экспорта (по умолчанию)

- 1080x1920 (9:16)
- 30 fps
- H.264 (`libx264`)
- AAC
- `-b:v 5000k`

---

## RU: Умный вертикальный кроп по лицу (smart face crop)

Иногда центр-кроп режет говорящего. Можно включить детекцию лица и сместить кроп так, чтобы лицо было ближе к центру.

### Включение

Для перерендеринга:

```bash
python3 rerender_videos.py --model gemma3 --smart-crop-face
```

Для обычного резака:

```bash
python -m podcast_reels_forge.scripts.video_processor \
  --input input/video.mp4 \
  --moments output/gemma3/moments.json \
  --outdir output/gemma3 \
  --vertical --smart-crop-face
```

Примечания:
- Используется OpenCV Haar cascade (быстро и оффлайн), но детекция может ошибаться.
- Если лицо не найдено (или OpenCV не установлен) — автоматически используется обычный центр-кроп.
- Можно подкрутить: `--face-samples` (сколько кадров смотреть) и `--face-min-size`.

---

## RU: Настройка через config.yaml

Основные блоки:
- `transcription.*` — качество/скорость транскрибации
- `ollama.*` — модели, таймауты, watchdog
- `video.*` — нарезка и базовые параметры FFmpeg
- `subtitles.*` — вшитые субтитры через `pycaps`; ключ `subtitles.font` по умолчанию указывает на `assets/fonts/bignoodletoooblique.ttf`
- `cache.*` — пропуск стадий, если результаты уже есть

Если некоторые модели Ollama долго «думают» до первого токена, используйте `ollama.model_overrides.*` (уже настроено в дефолтном `config.yaml`).

---

## RU: Частые проблемы

### Ollama «висит» или долго не отдаёт первый токен

- Увеличьте `ollama.watchdog.first_token_timeout` или задайте override в `ollama.model_overrides` для конкретной модели.

### Транскрибация падает по CUDA OOM

- Поставьте модель поменьше (`transcription.model`)
- Переключитесь на CPU (`transcription.device: cpu`)

### Видео получилось горизонтальным

- Для полного пайплайна проверьте `video.vertical_crop: true` в `config.yaml`
- Для гарантированного 9:16 и фиксированных параметров используйте `python3 rerender_videos.py`

### Сабы не вшиваются

- Если в ошибке фигурирует Chromium или Playwright, выполните `playwright install chromium`
- Проверьте, что `subtitles.font` указывает на существующий `.ttf`/`.otf` файл

---

## EN: What it is

Podcast Reels Forge is a tool that finds “viral moments” in long-form video/podcasts and cuts vertical clips for Reels/Shorts/TikTok.

Default pipeline:
1. Transcription via `faster-whisper` (local, CUDA/CPU)
2. Moment mining via an LLM (default: Ollama)
3. Cutting via FFmpeg (optional 9:16)

Key points:
- Uses 5 supported models (see `config.yaml`).
- Outputs are separated per model under `output/<model>/`.
- If one model fails/stalls, others continue.

---

## EN: Re-render video from existing moments.json (no LLM)

```bash
python3 rerender_videos.py --model gemma3
```

Defaults:
- writes into `output/<model>/reels_rerendered/`
- creates a `reel_XX.md` next to each clip with a description and 5 hashtags
- never overwrites: if a file exists, it creates `*_2`, `*_3`, ...

To overwrite:

```bash
python3 rerender_videos.py --model gemma3 --replace
```

---

## EN: Face-aware smart crop

```bash
python3 rerender_videos.py --model gemma3 --smart-crop-face
```

Notes:
- Uses OpenCV Haar cascade (fast, offline).
- Falls back to center-crop when no face detected.
