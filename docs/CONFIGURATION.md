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

## Article / Лонгрид по эпизоду

```yaml
article:
  enabled: true
  chunk_seconds: 600          # transcript window per request
  max_chars_chunk: 6000
  temperature: 0.2
  timeout: 900
  max_novel_word_ratio: 0.15  # rewritten in the model's own words
  max_length_ratio: 1.15      # padded
  min_length_ratio: 0.25      # abridged
  min_source_coverage: 0.45   # source vocabulary that must survive
```

RU: После вычитки gemma4 приводит транскрипт в вид читаемой статьи: разделы по
смыслу с заголовками, абзацы, исправленные ошибки. **Это не пересказ.** Слова,
обороты и лицо автора («я», «мы») сохраняются дословно; убираются только
слова-паразиты, оговорки, самоперебивы и дословные повторы.

Модель отвечает готовым markdown, а не JSON: текст почти дословный, длинный и
полон кавычек и тире — JSON-экранирование на нём регулярно ломалось (в одном
прогоне в текст статьи утёк литерал `paragraphs [`). У заголовков и пустых строк
экранировать нечего.

Три проверки ловят три разных способа отклониться:

- `max_novel_word_ratio` — доля слов, которых нет в исходном фрагменте. Выше
  порога означает, что текст **переписан своими словами**, а не отредактирован.
- `max_length_ratio` — текст **дописан**.
- `min_length_ratio` вместе с `min_source_coverage` — текст **сокращён**:
  ужался или растерял лексику источника.

Пороги откалиброваны по эталонной ручной вычитке реального 72-минутного эпизода:
у неё 3% новых слов, сохранено 64% лексики источника, длина 42% от исходной
(уходят слова-паразиты). Пересказ того же эпизода от третьего лица дал 24% новых
слов — именно это пороги и обязаны отсекать.

Нарушение — повтор запроса при `temperature=0` с явным напоминанием. Если и он
не прошёл, фрагмент сохраняется, но помечается в `<имя>.article.json`
(`chunks_flagged`, `faithfulness[].reasons`), чтобы непроверенный кусок не
выдавался за проверенный.

Результат: `<имя>.article.md` (для чтения) и `<имя>.article.json` (разделы,
тайминги, метаданные проверок). Транскрипт не изменяется.

EN: After proofreading, gemma4 edits the transcript into a readable article:
meaning-based sections with headings, paragraphs, corrected errors. **This is not
a retelling.** The author's words, phrasing and grammatical person are kept
verbatim; only filler, slips, self-interruptions and verbatim repetitions go.

The model answers in finished markdown rather than JSON: the text is
near-verbatim, long and full of quotes and dashes, and JSON escaping of that kept
breaking (one run leaked a literal `paragraphs [` into the prose). Headings and
blank lines have nothing to escape.

Three guardrails catch three different ways to drift:

- `max_novel_word_ratio` — share of words absent from the source fragment. Above
  the threshold the text was **rewritten**, not edited.
- `max_length_ratio` — the text was **padded**.
- `min_length_ratio` together with `min_source_coverage` — the text was
  **abridged**: it shrank or lost the source's vocabulary.

The thresholds are calibrated against a hand-approved reference edit of a real
72-minute episode: 3% new words, 64% of the source vocabulary kept, 42% of the
original length (spoken filler is what disappears). A third-person retelling of
the same episode scored 24% new words — exactly what these numbers must catch.

A violation triggers one retry at `temperature=0` with the constraint restated.
If that also fails the fragment is kept but flagged in `<stem>.article.json`
(`chunks_flagged`, `faithfulness[].reasons`), so an unverified passage is never
presented as verified.

Output: `<stem>.article.md` (to read) and `<stem>.article.json` (sections,
timings, guardrail metadata). The transcript is left untouched.

The model is selected via `llama_cpp.roles.article` (falls back to the
`cleanup_refine` model when omitted).

### Speakers / Разбивка по спикерам

RU: Если включена диаризация и рядом лежит `diarization.json`, лонгрид
собирается с разбивкой по говорящим: каждая реплика начинается с имени.

Единица текста в этом режиме — реплика, а не сегмент Whisper: Whisper режет речь
по паузам, и один его сегмент на 36 секунд запросто содержит троих. Поэтому
спикер назначается каждому слову (по пословным таймингам), а границы реплик
подтягиваются к концу предложения — иначе реплика начиналась бы с середины
фразы. Текст при этом берётся вычитанный, а не из списка слов: после вычитки
слова хранят исходное написание ASR, и сборка реплик из них отменила бы стадию
вычитки.

Имена подставляются из самого разговора: модель читает начало эпизода, где люди
представляются и обращаются друг к другу, и сопоставляет `SPEAKER_00` с именем.
Метка, для которой имя нигде не названо, остаётся технической — «Ведущий» и
«Гость» не выдумываются.

EN: With diarization enabled and a `diarization.json` alongside, the long-read is
built with speaker separation: every turn starts with a name.

The unit here is a turn, not a Whisper segment: Whisper splits on pauses, and one
36-second segment of its output happily holds three people. So the speaker is
assigned per word (from the word timings) and turn boundaries are nudged onto
sentence ends — otherwise a turn would start mid-phrase. The text comes from the
proofread segment rather than the word list: after proofreading those words still
carry the raw ASR spelling, and rebuilding turns from them would undo the
proofreading stage.

Names come from the conversation itself: the model reads the opening, where
people introduce and address each other, and maps `SPEAKER_00` to a name. A label
whose name is never stated keeps its technical id — no invented "Host" or "Guest".

> RU: pyannote склонен дробить голоса: на реальном эпизоде с тремя участниками
> он выделил пять. Если число участников известно, задайте
> `diarization.num_speakers` — это заметно улучшает разбивку.
>
> EN: pyannote tends to over-split: on a real three-person episode it found five
> speakers. When the count is known, set `diarization.num_speakers` — it
> noticeably improves the split.

## Running single stages / Запуск отдельных этапов

```bash
python3 start_forge.py --list-stages          # transcribe diarize proofread article analyze cut
python3 start_forge.py --only article         # only the long-read stage
python3 start_forge.py --only proofread,article
python3 start_forge.py --skip cut             # everything but video cutting
```

RU: `--only` и `--skip` принимают имена этапов через запятую. Неизвестное имя —
ошибка, а не молчаливый пропуск. Пропуск этапа не ломает остальные: если
вычитанный транскрипт остался от прошлого запуска, следующие этапы возьмут
именно его, а не сырой.

EN: `--only` and `--skip` take comma-separated stage names. An unknown name is an
error rather than a silent skip. Skipping a stage does not strand the others: a
proofread transcript left by an earlier run is what the later stages pick up.

В GUI те же этапы выбираются галочками на главной странице — страницы статические
и ничего не запускают сами, поэтому там собирается готовая команда для терминала.
/ The GUI offers the same choice as checkboxes on the dashboard; the pages are
static and run nothing themselves, so they assemble the command for you.

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
  font_size_px: 96        # fallback only — used when no .ass style file exists
  wrap_words: true
  max_lines: 2
  max_width_ratio: 0.65   # share of the frame width; drives chars per line
  vertical_offset: 0.0    # shift in frame heights, applied on top of the style MarginV
  fade_in_duration: 0.18  # \fad in seconds; 0 disables the fade
  fade_out_duration: 0.12
```

What each knob actually does:

- `max_width_ratio` sets the line length. The BBC guideline of 25 chars/line for
  9:16 assumes text spanning 0.65 of the frame, so the value scales from there:
  0.65 keeps 25 chars, 0.9 gives ~35.
- `vertical_offset` nudges the cue away from the edge it is anchored to, as a
  fraction of frame height, on top of the `MarginV` baked into the `.ass` style.
  `0.0` leaves the position entirely to the style editor.
- `fade_in_duration` / `fade_out_duration` emit an ASS `\fad` tag. If the two
  together exceed a cue's length they are scaled down proportionally so the cue
  still reaches full opacity.
- `font_size_px` only applies when no `.ass` style file is found; otherwise the
  size comes from the style editor.

`word_x_space` / `word_y_space` are legacy no-ops: word and letter spacing come
from the `.ass` style (`Spacing` in the editor). They are still parsed so old
configs keep loading, but they no longer appear in the GUI or in exported config.

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
