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

## llama.cpp / llama.cpp

```yaml
llama_cpp:
  url: "http://127.0.0.1:8080/v1/chat/completions"
  service:
    auto_start: true
    model_path: "models/gemma4_27b/gemma4-27b-q4_k_m.gguf"
    startup_timeout: 90
    n_gpu_layers: 32   # partial offload: 27b (Q4_K_M) doesn't fit fully in 16GB VRAM
    ctx_size: 4096
    batch_size: 1024
    ubatch_size: 512
    threads: 8
    main_gpu: 0
    extra_args: []
  roles:
    scout: "gemma4:27b"
    cleanup: "gemma4:27b"
    refine: "gemma4:27b"
    judge: "gemma4:27b"
    metadata: "gemma4:27b"
    proofread: "gemma4:27b"   # optional; defaults to the cleanup model
  timeout: 600
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
- The default workflow is llama.cpp-only and Gemma 4-only (currently the `gemma4:27b` lineup, partially offloaded to fit 16GB VRAM).

## Proofread / Вычитка транскрипта

```yaml
proofread:
  enabled: true
  max_chars_chunk: 4000   # max source chars per LLM request
  temperature: 0.0
  timeout: 600
  min_similarity: 0.8     # reject corrections below this similarity (0..1)
```

RU: После транскрибации (и диаризации) gemma4 вычитывает транскрипт: исправляет
орфографию, пунктуацию и регистр по правилам языка. Guardrail сравнивает каждое
исправление с оригиналом по нормализованному буквенному составу (без пунктуации
и регистра): если модель дописала, удалила или пересказала текст, правка
отклоняется и остаётся оригинал. Результат пишется в `<имя>.proofread.json` и
`<имя>.proofread.srt`; исходный транскрипт не изменяется. Дальше по конвейеру
(анализ, прожиг субтитров) используется вычитанная версия.

EN: After transcription (and diarization) gemma4 proofreads the transcript:
fixes spelling, punctuation and capitalization. A guardrail compares every
correction against the original by normalized letter content (punctuation and
case stripped): if the model added, removed or paraphrased anything, the
correction is rejected and the original text is kept. Output goes to
`<stem>.proofread.json` + `<stem>.proofread.srt`; the raw transcript is left
untouched. Downstream stages (analysis, burned subtitles) use the corrected
version.

The model is selected via `llama_cpp.roles.proofread` (falls back to the
`cleanup_refine` model when omitted).

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
  ass_style: "assets/subtitles/forge_subtitles.ass"
  font_size_px: 96
  wrap_words: true
  max_lines: 2
  max_width_ratio: 0.65
  vertical_offset: 0.0
  word_x_space: 6
  word_y_space: 8
  fade_in_duration: 0.18
  fade_out_duration: 0.12
```

Subtitles are rendered as **ASS** (Advanced SubStation Alpha) and burned in with
ffmpeg's `ass` filter. The visual style lives in the `.ass` file referenced by
`ass_style`; edit it through the GUI (Subtitles tab) or the standalone
`assets/subtitles/style-editor.html`.

The subtitle pipeline prefers `sentences` from the transcript JSON when available, then falls back to segment slicing.

## Diarization / Диаризация

```yaml
diarization:
  enabled: false
  model: "pyannote/speaker-diarization"
  # Exact number of speakers, if known (curbs over-clustering on noise/overlap).
  # Leave empty/null to let pyannote estimate it automatically.
  num_speakers: null
```

## Examples / Примеры

Minimal local-first config:

```yaml
transcription:
  model: "large-v3"
  device: "auto"
  language: "auto"

llama_cpp:
  roles:
    scout: "gemma4"
    cleanup: "gemma4"
    refine: "gemma4"
    judge: "gemma4"

processing:
  reels_count: 3
  reel_padding: 5
```

Compatibility note: older `models:` lists are still accepted by the loader, but the new role mapping is the preferred format.
