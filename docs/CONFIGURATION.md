# Конфигурация / Configuration

## Структура config.yaml

Файл `config.yaml` — центральное место для всех настроек пайплайна.

The `config.yaml` file is the single source of truth for all pipeline settings.

### paths — Пути к директориям

paths — Directory paths

```yaml
paths:
  input_dir: "input"    # Папка с исходными файлами / Folder with source files
  output_dir: "output"  # Папка для результатов / Folder for results
```

### cli — Настройки командной строки

cli — CLI settings

```yaml
cli:
  quiet: false   # true = только ошибки / errors only
  verbose: false # true = подробный вывод / verbose output
```

### transcription — Транскрипция

transcription — Transcription

```yaml
transcription:
  model: "small"               # Модель Whisper / Whisper model
  device: "cuda"               # cuda | cpu
  language: "ru"               # ru | en | auto
  beam_size: 5                 # Размер луча / Beam size
  compute_type: "int8_float16" # Тип вычислений / Compute type
```

**Доступные модели / Available models:**

| Модель / Model | Параметры / Params | VRAM | Качество / Quality |
| -------- | ----------- | ------ | ---------- |
| tiny | 39M | ~1GB | Низкое |
| base | 74M | ~1GB | Базовое |
| small | 244M | ~2GB | Среднее |
| medium | 769M | ~5GB | Хорошее |
| large-v3 | 1550M | ~10GB | Лучшее |

**Типы вычислений / Compute types:**

| Тип / Type | Скорость / Speed | Точность / Accuracy | GPU поддержка / GPU support |
| ----- | ---------- | ---------- | --------------- |
| float32 | Медленно | Высокая | Все |
| float16 | Быстро | Высокая | SM >= 7.0 |
| int8_float16 | Очень быстро | Хорошая | SM >= 7.0 |
| int8 | Очень быстро | Средняя | Все |

### llm — Провайдер LLM

llm — LLM provider

```yaml
llm:
  provider: "ollama"   # ollama | openai | anthropic | gemini
  
  # Модели для облачных провайдеров / Models for cloud providers
  openai_model: "gpt-4o-mini"
  anthropic_model: "claude-3-5-sonnet-20241022"
  gemini_model: "gemini-1.5-flash"
```

### ollama — Настройки Ollama

ollama — Ollama settings

```yaml
ollama:
  url: "http://127.0.0.1:11434/api/generate"
  model: "gemma2:9b"
  timeout: 900           # Таймаут запроса (сек) / Request timeout (sec)
  temperature: 0.3       # Креативность (0.0 - 1.0) / Creativity (0.0 - 1.0)
  chunk_seconds: 600     # Размер чанка (сек) / Chunk size (sec)
  max_chars_chunk: 12000 # Макс. символов / Max chars

  # Watchdog: если генерация "зависла" или слишком медленная, запрос будет прерван
  # и выполнен повторно (с возможной сменой модели).
  watchdog:
    enabled: true
    first_token_timeout: 120  # сек без какого-либо вывода до первого токена
    stall_timeout: 120        # сек без вывода во время стриминга
    log_interval: 10          # каждые N сек логируется прогресс
    max_retries: 2            # сколько раз ретраить при зависании/таймауте

  # Список моделей, которые пробуются по очереди, если основная слишком медленная.
  # Важно: модели должны быть установлены в Ollama (ollama pull ...)
  fallback_models: []
```

### prompts — Промпты

prompts — Prompts

```yaml
prompts:
  language: "ru"        # ru | en | auto
  variant: "default"    # default | a | b
```

### processing — Обработка

processing — Processing

```yaml
processing:
  clips:
    stories:
      count: 2       # Количество сторис (до 15с) / Stories count (up to 15s)
    reels:
      count: 3       # Количество рилс (до 1м) / Reels count (up to 1m)
    long_reels:
      count: 1       # Длинные рилс (до 3м) / Long reels (up to 3m)
    highlights:
      count: 1       # Роликов с хайлайтами / Highlights videos
      moments_count: 5 # Моментов в хайлайте / Moments in highlight

  reels_count: 4          # [Legacy] Общее количество клипов / Total clips count
  reel_min_duration: 30   # Мин. длина (сек) / Min length (sec)
  reel_max_duration: 60   # Макс. длина (сек) / Max length (sec)
  reel_padding: 0         # Отступ (сек) / Padding (sec)
```

### exports — Экспорт форматов

exports — Export formats

```yaml
exports:
  webm: false       # Экспорт в WebM / Export WebM
  gif: false        # Экспорт в GIF / Export GIF
  audio_only: false # Экспорт только аудио / Export audio only
```

### video — Настройки видео

video — Video settings

```yaml
video:
  threads: 4              # Потоки FFmpeg / FFmpeg threads
  vertical_crop: true     # Кроп 16:9 → 9:16 / Crop 16:9 → 9:16
  video_bitrate: "5M"     # Битрейт видео / Video bitrate
  audio_bitrate: "192k"   # Битрейт аудио / Audio bitrate
  preset: "fast"          # Пресет / Preset
```

### diarization — Диаризация

diarization — Diarization

```yaml
diarization:
  enabled: false
  model: "pyannote/speaker-diarization"
```

**Требования для диаризации / Diarization requirements:**

1. Установить `pyannote-audio`:

  ```bash
  pip install pyannote-audio
  ```

1. Получить токен на Hugging Face и установить:

  ```bash
  export PYANNOTE_TOKEN="hf_..."
  ```

---

## Переменные окружения / Environment variables

| Переменная / Variable | Описание / Description |
| ------------ | ---------- |
| `OPENAI_API_KEY` | API ключ OpenAI |
| `ANTHROPIC_API_KEY` | API ключ Anthropic |
| `GEMINI_API_KEY` | API ключ Google Gemini |
| `PYANNOTE_TOKEN` | Токен Hugging Face для pyannote |
| `WHISPER_VENV_ACTIVE` | Флаг активации venv (внутренний) |

---

## Примеры конфигураций / Configuration examples

### Минимальная конфигурация (Ollama)

Minimal configuration (Ollama)

```yaml
paths:
  input_dir: "input"
  output_dir: "output"

transcription:
  model: "small"
  device: "cuda"
  language: "auto"

llm:
  provider: "ollama"

ollama:
  model: "gemma2:9b"

processing:
  reels_count: 4
```

### Конфигурация для OpenAI

Configuration for OpenAI

```yaml
paths:
  input_dir: "input"
  output_dir: "output"

transcription:
  model: "medium"
  device: "cuda"
  language: "auto"

llm:
  provider: "openai"
  openai_model: "gpt-4o"

processing:
  reels_count: 6
  reel_min_duration: 20
  reel_max_duration: 90

video:
  vertical_crop: true

exports:
  webm: true
```

### Конфигурация для слабого GPU

Configuration for weak GPU / CPU

```yaml
transcription:
  model: "tiny"
  device: "cpu"
  compute_type: "float32"

ollama:
  model: "gemma2:2b"
  timeout: 1800

video:
  threads: 2
  preset: "ultrafast"
```
