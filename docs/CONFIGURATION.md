# Конфигурация / Configuration

`config.yaml` is the single source of truth for the local-only pipeline.

## Paths / Пути

```yaml
paths:
  input_dir: "input"
  output_dir: "output"
```

## Transcription / Транскрипция

```yaml
transcription:
  model: "large-v3"
  device: "auto"        # auto | cuda | cpu
  language: "auto"      # auto | ru | en
  beam_size: 6
  compute_type: "auto"  # auto | float32 | float16 | int8 | int8_float16 | int8_float32
```

The transcription stage now writes additive fields such as `source_audio`, `timing_version`, `language_confidence`, `segments[].words`, and `sentences`.

## Ollama / Ollama

```yaml
ollama:
  url: "http://127.0.0.1:11434/api/generate"
  roles:
    scout: "gemma4:e4b"
    cleanup: "gemma3:4b"
    refine: "gemma3:12b"
    judge: "gemma4:26b"
    metadata: "gemma4:26b"
  timeout: 420
  temperature: 0.2
  chunk_seconds: 900
  max_chars_chunk: 12000
  watchdog:
    enabled: true
    first_token_timeout: 60
    stall_timeout: 90
    log_interval: 10
    max_retries: 1
  fallback_models: []
  role_overrides:
    scout:
      timeout: 360
      chunk_seconds: 1200
      temperature: 0.35
  model_overrides: {}
```

Notes:

- `roles` is the default way to configure the staged analysis pipeline.
- `model_overrides` is retained only for legacy compatibility.
- The default workflow is Ollama-only and Gemma-only.

## Processing / Обработка

```yaml
processing:
  quality_filters:
    min_score: 7
    min_duration: 15
    max_duration: 180
    face_min_ratio: 0.3
  clips:
    stories:
      count: 2
      max_duration: 15
    reels:
      count: 3
      max_duration: 60
    long_reels:
      count: 1
      max_duration: 180
    highlights:
      count: 1
      moments_count: 5
  reels_count: 3
  reel_min_duration: 30
  reel_max_duration: 60
  reel_padding: 5
```

## Video / Видео

```yaml
video:
  threads: 4
  vertical_crop: true
  smart_crop_face: true
  video_bitrate: "6M"
  audio_bitrate: "192k"
  preset: "fast"
  use_nvenc: true
  face_samples: 9
  face_min_size: 72
```

Face crop now uses multiple samples and median smoothing before FFmpeg renders the final crop.

## Subtitles / Субтитры

```yaml
subtitles:
  enabled: true
  font: "assets/fonts/bignoodletoooblique.ttf"
  css: "assets/subtitles/forge_subtitles.css"
  font_size_px: 46
  wrap_words: true
  max_lines: 2
  max_width_ratio: 0.92
  vertical_align: "bottom"
  vertical_offset: 0.0
```

The subtitle pipeline prefers `sentences` from the transcript JSON when available, then falls back to segment slicing.

## Diarization / Диаризация

```yaml
diarization:
  enabled: false
  model: "pyannote/speaker-diarization"
```

## Examples / Примеры

Minimal local-first config:

```yaml
transcription:
  model: "large-v3"
  device: "auto"
  language: "auto"

ollama:
  roles:
    scout: "gemma4:e4b"
    cleanup: "gemma3:4b"
    refine: "gemma3:12b"
    judge: "gemma4:26b"

processing:
  reels_count: 3
  reel_padding: 5
```

Compatibility note: older `models:` lists are still accepted by the loader, but the new role mapping is the preferred format.
