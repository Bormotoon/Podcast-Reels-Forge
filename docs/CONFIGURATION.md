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

## Analysis quality / Качество отбора моментов

RU: Блок `processing.analysis` настраивает то, как из транскрипта выбираются
моменты. Он целиком опционален — у каждого ключа есть значение по умолчанию
в коде, поэтому блок можно не писать вовсе.

EN: `processing.analysis` tunes how moments are picked out of the transcript.
The whole block is optional — every key has a code default, so it can be
omitted entirely.

```yaml
processing:
  analysis:
    cleanup_cap: 16          # candidates per cleanup request (batched above that)
    json_retry: 1            # re-asks when a reply does not parse as JSON
    strict_json_schema: true # send the full moments schema as a sampling grammar
    validation:
      chunk_tolerance_s: 3.0            # allowed drift past a chunk's own bounds
      require_candidate_overlap: true   # drop cleanup/judge output that matches no input
    quote_verification:
      enabled: true
      min_ratio: 0.55        # below this the quote counts as invented
      refine_boundaries: true
    boundary_snap:
      enabled: true
      max_shift_s: 3.0       # how far a bound may move onto a speech boundary
    judge_context:
      enabled: true          # show the judge the clip's real opening and ending
      max_candidates: 14
      head_seconds: 15
      tail_seconds: 5
      max_excerpt_chars: 260
    episode_context:
      enabled: true          # one LLM call summarizing the episode for the scout
      max_digest_chars: 4000
    audio_features:
      enabled: true          # measure loudness and pauses with ffmpeg
      timeout_s: 30
      silence_noise_db: -30.0
      silence_min_s: 0.35
    scoring:
      weights: {}            # override any of the priority factors
    diversity:
      enabled: true          # avoid several final clips about the same thing
      max_topic_similarity: 0.5
```

### score и priority / score vs priority

RU: `score` — оценка модели по шкале 1-10. Именно с ней сравнивается
`processing.quality_filters.min_score` на стадии нарезки. `priority` —
отдельное поле: комбинированное значение эвристик, по которому кандидаты
ранжируются между собой. Оба попадают в `moments.json` и `reels.md`.

EN: `score` is the model's own 1-10 rating, and it is what
`processing.quality_filters.min_score` compares against at the cut stage.
`priority` is a separate field: the combined heuristic value candidates are
ranked by. Both appear in `moments.json` and `reels.md`.

### Веса скоринга / Scoring weights

RU: Ключи `scoring.weights` (значения по умолчанию в скобках): `base` (0.55,
вклад оценки модели), `hook` (1.8), `readability` (1.2), `completeness`
(1.0), `speaker` (0.6), `duration` (1.4), `quote` (0.8, подтверждённость
цитаты), `audio` (0.7), `speech_rate` (0.4), `mid_thought` (1.0 — штраф за
обрыв на полуслове, вычитается). Неизвестные и нечисловые ключи молча
игнорируются.

EN: `scoring.weights` keys (defaults in brackets): `base` (0.55, the model's
own score), `hook` (1.8), `readability` (1.2), `completeness` (1.0),
`speaker` (0.6), `duration` (1.4), `quote` (0.8, how well the quote is
grounded), `audio` (0.7), `speech_rate` (0.4), `mid_thought` (1.0 — the
penalty for cutting mid-thought, subtracted). Unknown or non-numeric keys are
ignored.

RU: Факторы, которые не удалось измерить (нет исходного аудио, нет пословных
таймкодов), берут нейтральное значение 0.5, чтобы кандидаты с сигналом и без
него оставались сравнимыми.

EN: Factors that could not be measured (no source audio, no word timings) use
a neutral 0.5, so candidates with and without the signal stay comparable.

> RU: GUI пересобирает `config.yaml` из шаблона. Блок `analysis` записывается
> в него значениями по умолчанию, поэтому изменённые вручную значения будут
> сброшены при сохранении настроек из интерфейса.
>
> EN: The GUI regenerates `config.yaml` from a template. The `analysis` block
> is written out with its defaults, so hand-edited values here are reset when
> settings are saved from the interface.

## Processing / Обработка

```yaml
processing:
  quality_filters:
    min_score: 7
    min_duration: 15
    max_duration: 180
    face_min_ratio: 0.3
  clips_per_hour: 10   # clips per hour of total runtime; 0 = fixed counts
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

### clips_per_hour

RU: Целевое число клипов считается от **суммарного** хронометража эпизода:
`round(длительность_в_часах × clips_per_hour)`. Эпизод на 1.5 часа при
`clips_per_hour: 10` даст 15 клипов. Счётчики `count` в `clips` при этом
задают только пропорции типов (2:3:1:5 по умолчанию), а не абсолютные
количества; распределение — методом наибольших остатков, сумма сходится
точно. `0` выключает масштабирование — работают фиксированные количества.

EN: The target clip count is computed from the episode's **total** runtime:
`round(hours × clips_per_hour)`. A 1.5-hour episode at `clips_per_hour: 10`
yields 15 clips. The `count` values under `clips` then only set the type mix
(2:3:1:5 by default), not absolute counts; apportionment uses the
largest-remainder method so the sum matches exactly. `0` disables scaling and
the fixed counts apply.

RU: Когда целевое число превышает вместимость одного промпта, cleanup и judge
автоматически работают несколькими запросами (батчами по `cleanup_cap` и
`judge_context.max_candidates` кандидатов); сравнение внутри judge становится
побатчевым, финальное детерминированное ранжирование остаётся глобальным.

EN: When the target exceeds what fits into one prompt, cleanup and judge
automatically run as multiple requests (batches of `cleanup_cap` and
`judge_context.max_candidates` candidates); judge comparison becomes
per-batch while the final deterministic ranking stays global.

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
