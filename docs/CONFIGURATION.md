# Конфигурация / Configuration

`config.yaml` is the single source of truth for the local-only pipeline.

## Paths / Пути

```yaml
paths:
  input_dir: "input"
  output_dir: "output"
```

## YouTube / Загрузка с YouTube

RU: Стадия `fetch` — первая в конвейере. Ролик по ссылке, плейлист целиком или
весь канал скачиваются в `download_dir`, после чего файл ничем не отличается от
положенного руками. Нужен `yt-dlp` (`pip install -U yt-dlp`).

EN: The `fetch` stage comes first in the pipeline. A video link, a whole playlist
or an entire channel is downloaded into `download_dir`, after which the file is
indistinguishable from one placed by hand. Needs `yt-dlp`.

```yaml
youtube:
  sources: []                  # постоянные источники: ссылки, @handle
  exclude: []                  # никогда не брать: обычно плейлист
  download_dir: "input/youtube"
  api_key_env: "YOUTUBE_API_KEY"
  download: "audio"            # audio | auto | video
  max_height: 1080
  filename_template: "%(upload_date>%Y-%m-%d)s - %(title).150B [%(id)s].%(ext)s"
  archive: "input/youtube/.archive.txt"
  limit: 0                     # 0 = без ограничения, отсчёт от новых
  since: null                  # "2026-01-01"
  until: null
  min_duration: 60             # отсекает Shorts
  max_duration: 0
  skip_live: true
  cookies_file: null
  retries: 3
  rate_limit: null             # напр. "5M"
  ydl_options: {}              # сырые опции yt-dlp, подмешиваются последними
```

| Ключ | Что делает |
|---|---|
| `sources` | Источники, обрабатываемые каждым запуском. Разовые задаются флагом `--youtube`, он идёт первым в списке. |
| `exclude` | Ролики, которые не берутся никогда — ни скачиваются, ни обрабатываются. Принимает те же формы, что и `sources`; плейлист удобнее перечня id: правило «всё с канала, кроме этого шоу» остаётся верным само, когда в шоу добавляют выпуск. Исключение сильнее прямой ссылки и применяется **до** `limit`, поэтому `--yt-limit 5` даёт пять подходящих роликов, а не «пять новых, из которых часть выпала». Разовые исключения (`--yt-exclude`) складываются с этим списком, а не заменяют его. |
| `download` | `audio` (по умолчанию) — всегда только аудиодорожка: час подкаста ~64 МБ вместо ~1 ГБ. Нарезать нечего, и стадия `cut` для таких эпизодов сообщит, что видео нет. `video` — всегда видео. `auto` — видео, только если в наборе стадий есть `cut`; под `--only fetch` намерение неизвестно, и `auto` оставляет видео, потому что ошибка в другую сторону стоит перекачки всего канала. Разовое переопределение — `--yt-audio-only` / `--yt-video`. |
| `max_height` | На выходе всё равно вертикальный клип шириной 1080, поэтому 4K-исходник — просто занятый диск. `0` — брать лучшее доступное. |
| `filename_template` | Шаблон вывода в синтаксисе yt-dlp. Дата ставит папки эпизодов в хронологический порядок, id делает имя уникальным и позволяет узнать уже скачанный ролик после переименования на YouTube. `.150B` — предел в **байтах**: кириллица весит два байта на символ, а лимит имени файла 255. |
| `archive` | Журнал взятых id: ночной прогон канала берёт только новое. `null` — выключить. |
| `min_duration` | Нижняя граница длительности. 60 отсекает Shorts. Ролики с неизвестной длительностью (так бывает при перечислении без API-ключа) фильтр пропускает: иначе прогон канала без ключа молча остался бы пустым. |
| `cookies_file` | Файл cookies в формате Netscape — для возрастных ограничений и материалов для спонсоров. |
| `ydl_options` | Опции yt-dlp как есть, подмешиваются последними и перекрывают всё вычисленное. YouTube регулярно меняет отдачу видео, и чинится это обычно одной опцией — знать её здесь важнее, чем править код. |

RU: Если ролик не скачался, стадия делает вторую попытку через другой набор
клиентов YouTube (`web_safari`, `tv`, `android`). Это лечит реальный случай: на
части старых роликов дефолтные клиенты не видят ни одного формата и ролик
выглядит как «This video is not available», хотя API отдаёт его публичным и без
региональных ограничений. Повтор только после неудачи — навязывать этот набор
всем подряд значит менять рабочее на неизвестное: у клиента `tv`, например,
часть форматов приходит под DRM. Если `player_client` задан вручную через
`ydl_options`, повтора нет: иначе заданная настройка выглядела бы бесполезной.

EN: When a download fails, the stage makes one more attempt through another set
of YouTube clients (`web_safari`, `tv`, `android`). This fixes a real case: for
some older videos the default clients see no formats at all and the video reads
as "This video is not available", even though the API reports it public with no
region restriction. The retry happens only after a failure — forcing that set on
everything would trade what works for the unknown (the `tv` client returns some
formats DRM-protected). A `player_client` set by hand in `ydl_options` disables
the retry, or that setting would look like it did nothing.

RU: Чего повтор не лечит: блокировку правообладателем и региональные
ограничения. Такой ролик недоступен любому клиенту, и стадия честно скажет об
этом, не прерывая остальную очередь.

EN: What the retry cannot fix: a rights-holder block or a region lock. Such a
video is out of reach for every client, and the stage says so plainly without
stopping the rest of the queue.

RU: Ключ `YOUTUBE_API_KEY` не обязателен — без него перечисление делает сам
yt-dlp. Он читается из `.env` в корне проекта или из окружения и уходит
заголовком `X-goog-api-key`, а не в строке запроса: под `--verbose` urllib3
логирует каждый URL, и ключ попал бы в лог. С ключом перечисление целого канала
стоит ~41 единицу из бесплатных 10 000 в сутки.

EN: `YOUTUBE_API_KEY` is optional — without it yt-dlp does the listing. It is read
from the project-root `.env` or the environment and travels as an
`X-goog-api-key` header rather than in the query string: under `--verbose`
urllib3 logs every URL, and the key would land in the log. With a key, listing a
whole channel costs ~41 units of the free 10,000/day quota.

RU: Как только источники заданы — флагом `--youtube` или списком `sources`, —
обрабатываются только названные ролики: иначе один запуск потянул бы за собой все
локальные эпизоды из `input/` с перекодированием аудио для каждого. Снимается
флагом `--yt-all-inputs`.

EN: Once sources are given — via `--youtube` or the `sources` list — only the
named videos are processed: otherwise one run would drag along every local
episode in `input/`, transcoding audio for each. `--yt-all-inputs` lifts the
restriction.

## Host memory / Освобождение памяти хоста

RU: Пайплайн держит на хосте несколько гигабайт (llama-server плюс питон с
torch). Если рядом живут виртуалки, этого хватает, чтобы упереться в OOM —
причём ядро убивает не пайплайн, а процесс с наибольшим `oom_score_adj`
(у snap-сборки VS Code он равен 300, то есть она сама вызывается на роль жертвы).

EN: The pipeline holds several gigabytes on the host. With VMs alongside, that is
enough to reach the OOM killer — which kills not the pipeline but whatever has
the highest `oom_score_adj` (a snap-packaged VS Code sets its own to 300).

```yaml
host_memory:
  enabled: false
  domains: []              # баллонить (пусто = все запущенные, кроме stop_domains)
  stop_domains: []         # выключать на время прогона и поднимать после
  shutdown_timeout_s: 180
  headroom_mb: 1536        # запас гостю сверх занятого им
  min_mb: 1024             # ниже не опускаться никогда
  state_file: ".forge-host-memory.json"
  connect: "qemu:///system"
```

| Ключ | Что делает |
|---|---|
| `domains` | У кого забирать неиспользуемую память баллоном, на живую. Размер считается от того, что гость сам сообщает занятым, а не от числа, вписанного полгода назад. |
| `stop_domains` | Кого выключать целиком. Возврат — обычный старт домена: чистая загрузка гостя, а не продолжение с того же места. |
| `headroom_mb` | Сколько оставить гостю сверх занятого. Сжимать впритык — верный способ получить OOM уже внутри ВМ. |

### Почему баллон, а не пауза / Why ballooning, not pausing

RU: `virsh suspend` не освобождает ни байта — останавливает vCPU, но qemu
продолжает держать всю память гостя. `managedsave` libvirt не даст выполнить для
домена с назначенным PCI-устройством.

EN: `virsh suspend` frees nothing — it stops the vCPUs while qemu keeps every
page mapped. `managedsave` is refused by libvirt for a domain with an assigned
PCI device.

### Ограничение: проброшенный PCI / The passthrough limit

RU: **На ВМ с проброшенным PCI-устройством баллон не освобождает ничего.** Вся
память такого гостя залочена в RAM (`VmLck` равен её размеру), потому что IOMMU
нужен постоянный маппинг для DMA. Баллон при этом честно сдвигается и гость
сообщает о свободных страницах, а RSS у qemu не падает ни на байт: память
отнимается у гостя и не достаётся никому. Такие домены определяются по `VmLck` и
пропускаются; забрать у них память может только `stop_domains`.

EN: **Ballooning frees nothing on a VM with a passed-through PCI device.** Its
whole guest memory is locked into RAM (`VmLck` equals its size) because the IOMMU
needs a permanent DMA mapping. The balloon does move and the guest does report
free pages, yet qemu's RSS does not drop by a byte — the memory is taken from the
guest and handed to nobody. Such domains are detected via `VmLck` and skipped;
only `stop_domains` can reclaim their memory.

### Возврат / Restoring

RU: Исходные размеры пишутся на диск (с fsync) ДО первого изменения. Обычный
выход, исключение, Ctrl+C и `kill` возвращают память сразу; после SIGKILL или OOM
её вернёт следующий запуск либо отдельная команда. Неудавшийся возврат сохраняет
запись, а не удаляет её, — иначе ВМ осталась бы сжатой, и никто бы об этом не
знал.

EN: Original sizes are fsynced to disk BEFORE the first change. A normal exit, an
exception, Ctrl+C and `kill` restore immediately; after a SIGKILL or an OOM the
next run or the standalone command does it. A failed restore keeps its record
rather than dropping it — otherwise a VM would stay shrunken with nothing
tracking it.

```bash
python3 -m podcast_reels_forge.scripts.host_memory --status    # раскладка, ничего не менять
python3 -m podcast_reels_forge.scripts.host_memory --restore   # вернуть всё вручную
python3 start_forge.py --free-ram                              # включить на один запуск
python3 start_forge.py --no-free-ram                           # не трогать ВМ на один запуск
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
    json_retry_budget: 10    # episode-wide cap on those re-asks
    strict_json_schema: true # send each stage's schema as a sampling grammar
    # chunk_overlap_s: 30    # default: chunk_seconds/8 clamped to 20..45 s
    validation:
      chunk_tolerance_s: 3.0            # allowed drift past a chunk's own bounds
      require_candidate_overlap: true   # drop cleanup/judge output traceable to no input
    quote_verification:
      enabled: true
      min_ratio: 0.55        # below this: rejected right after the scout
      min_final_ratio: 0.75  # below this: never enters the final cut
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
      max_candidates: 40     # probe only the best N (0 = all)
      parallelism: 4         # ffmpeg probes in parallel
      cache: true            # audio_features_cache.json in the model folder
    scoring:
      weights: {}            # override any of the priority factors
    diversity:
      enabled: true          # avoid several final clips about the same thing
      max_topic_similarity: 0.5
      mmr_lambda: 0.7        # MMR: quality - lambda * similarity to selected
    selection:
      max_overlap_ratio: 0.2 # allowed overlap, as a share of the shorter clip
```

### Цитата как доказательство / The quote as evidence

RU: Цитата — единственное поле, которое связывает кандидата с реально
сказанным. Она ищется в транскрипте сначала дословно (подряд, без учёта
регистра и пунктуации), затем ограниченным нечётким сравнением по словам.
Кандидат ниже `min_ratio` отбрасывается сразу после scout, ниже
`min_final_ratio` — перед финальным отбором. Всё отброшенное с причиной
лежит в `rejected_candidates.json`. В `moments.json` у каждого момента есть
`quote_match_method` (`exact`/`fuzzy`), `quote_start`/`quote_end`, а
финальная проверка гарантирует, что цитата целиком внутри клипа. Поля,
сгенерированные кодом, а не моделью (например caption), перечислены в
`derived_fields`. Если пословных таймингов нет вовсе (старый транскрипт без
`words`), текст фраз интерполируется по времени.

EN: The quote is the only field tying a candidate to what was actually said.
It is looked up verbatim first (contiguous, ignoring case and punctuation),
then with a bounded word-level fuzzy match. Below `min_ratio` a candidate is
rejected right after the scout; below `min_final_ratio` it may not enter the
final selection. Everything rejected is kept, with the reason, in
`rejected_candidates.json`. Each moment in `moments.json` carries
`quote_match_method` (`exact`/`fuzzy`) and `quote_start`/`quote_end`, and a
final check guarantees the quote lies inside the clip. Fields generated by
code rather than the model (e.g. the caption) are listed in `derived_fields`.
Without any word timings (an old transcript without `words`) sentence text is
interpolated over time.

### Метрики прогона / Run metrics

RU: `analysis_metrics.json` в папке модели: число кандидатов на каждом шаге,
выживаемость после quote-фильтра/cleanup/judge, доля точных совпадений
цитат, доля низкой уверенности, средний и p95 сдвиг границ, доля дублей,
тематическое разнообразие финала, заполнение квот, вызовы/время/объём
промптов каждой LLM-стадии и израсходованный бюджет повторов.

EN: `analysis_metrics.json` in the model folder: candidate counts at every
step, survival after the quote gate/cleanup/judge, exact quote match rate,
low-confidence rate, mean and p95 boundary shift, duplicate rate, topic
diversity of the final set, quota fill rate, calls/time/prompt size per LLM
stage, and the retry budget spent.

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

### Rare terms / Перепроверка редких слов

```yaml
proofread:
  terms:
    enabled: false      # network access — opt in
    min_occurrences: 3
    max_terms: 20
    pause_seconds: 1.0
```

RU: Распознавание стабильно ошибается на именах и прозвищах: «Курокрад»
превращается в «Курократ». Внутри эпизода это не поймать — правильной формы там
просто нет, — а спрашивать человека про каждое слово незачем.

Проверяется не «как правильно», а какое из написаний внешний источник **вообще
знает**: у «курокрад» в Викисловаре 9 совпадений, у «курократ» — ноль. Кандидаты
находятся по заглавной букве не в начале предложения, варианты написания
получаются заменой звуков, которые путает ASR (звонкие/глухие на конце, е/э,
о/а). Первая буква не меняется никогда: её распознавание почти не теряет, а
подмена даёт просто другое слово.

Правка вносится **только** когда исходное написание источнику неизвестно, а
вариант известен уверенно. Оба известны, оба неизвестны, сеть недоступна —
текст остаётся как есть. Неудачный запрос отличается от «ноль совпадений»:
сбой сети не должен читаться как «слово неизвестно» и переписывать текст.

Всё, что изменено, попадает в `<имя>.proofread.json` → `proofread.term_fixes`
вместе с доказательством, так что правки можно проверить.

**Приватность.** Это единственное место, где данные покидают машину. Наружу
уходит только подозрительное слово и максимум одно соседнее (контекст помогает:
клуб «Подземелье Деновалис» по одному второму слову не находится). Текст
расшифровки не отправляется. Источник — официальный API Викисловаря и
Википедии, без ключа; результаты кэшируются в `term_lookups.json`, поэтому слово
запрашивается один раз.

EN: Recognition reliably trips over names and nicknames: "Курокрад" comes back as
"Курократ". The episode cannot settle it — the correct form is simply absent —
and asking a human about every word defeats the point.

The check is not "what is correct" but which spelling an outside source **knows
at all**: Wiktionary has 9 hits for "курокрад" and none for "курократ".
Candidates are spotted by a capital letter mid-sentence; variants come from
swapping the sounds ASR confuses (final devoicing, е/э, о/а). The first letter is
never swapped — recognition rarely loses it, and changing it just yields a
different word.

A fix lands **only** when the original is unknown to the source and a variant is
known with confidence. Both known, neither known, or no network — the text stays
as it is. A failed lookup is kept distinct from "zero hits": a flaky network must
not read as "unknown word" and start rewriting.

Everything changed is recorded in `<stem>.proofread.json` →
`proofread.term_fixes`, with the evidence, so the edits can be audited.

**Privacy.** This is the one place where data leaves the machine. Only the
suspect word and at most one neighbouring word are sent (context matters: the
club "Подземелье Деновалис" is unfindable by its second word alone). No
transcript text is transmitted. The source is the official Wiktionary/Wikipedia
API, keyless; results are cached in `term_lookups.json`, so a word is looked up
once.

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
python3 start_forge.py --list-stages          # fetch transcribe diarize proofread article analyze cut
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

## Прогон без присмотра / Unattended runs

RU: Подробно — в [AUTONOMOUS.md](AUTONOMOUS.md). Все ключи опциональны.

EN: Details in [AUTONOMOUS.md](AUTONOMOUS.md). Every key is optional.

```yaml
autonomy:
  lock_file: ".forge.lock"   # one run at a time; a second one exits with 75
  scheduling: stage          # stage | episode
  log_dir: "logs"            # daily-rotated forge.log at INFO ("" disables)
  log_keep_days: 14
  runs_dir: ""               # run reports; default <output_dir>/_runs
  min_free_disk_gb: 5        # preflight refuses to start below this
  notify:
    when: failure            # always | failure | never
    command: ""              # gets FORGE_OUTCOME, FORGE_EXIT_CODE, FORGE_SUMMARY, FORGE_REPORT
    webhook: ""              # POST of the summary as JSON

audio:
  listening_copy: false             # 320k MP3 next to the source (models use the WAV)
  delete_wav_after_analysis: false  # drop the 16 kHz WAV once the episode is analysed

youtube:
  self_update: true          # pip install -U yt-dlp every self_update_days
  self_update_days: 7

proofread:
  scope: full                # full | clips (only the selected clips' spans; needs article off)

processing:
  analysis:
    llm_cache: true          # cache LLM answers in <model folder>/llm_cache
  quality_filters:
    render_rejected: false   # encode rejected clips into reels/rejected/ too

subtitles:
  keep_nosubs: false         # also render a clean reel_XX.nosubs.mp4

video:
  two_speaker_layout: split  # split | single
  qa: true                   # ffprobe check of every rendered clip
  qa_blackdetect: false      # also fail mostly-black clips (one more decode)
```

RU: Фильтры `processing.quality_filters` (`min_score`, `min_duration`,
`max_duration`) теперь применяются уже при отборе моментов, а не только при
нарезке: каждый выбранный слот достаётся клипу, который действительно будет
нарезан.

EN: `processing.quality_filters` (`min_score`, `min_duration`,
`max_duration`) are now enforced at selection, not only at the cut: every
selected slot goes to a clip that will actually be cut.
