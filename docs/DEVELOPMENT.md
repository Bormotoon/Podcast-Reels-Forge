# Руководство для разработчиков / Developer Guide

## Установка для разработки / Development setup

```bash
git clone https://github.com/Bormotoon/Podcast-Reels-Forge.git
cd Podcast-Reels-Forge

python3 -m venv whisper-env
source whisper-env/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pip install -e ".[youtube]"     # опционально / optional
```

RU: тесты не требуют GPU, llama-server, ffmpeg или сети — всё тяжёлое
подменяется. Машинные настройки держите в `config.local.yaml` (в `.gitignore`),
а не правками `config.yaml`.

EN: the tests need no GPU, llama-server, ffmpeg or network — everything heavy is
stubbed. Keep machine-specific settings in the gitignored `config.local.yaml`,
not as edits to `config.yaml`.

## Структура кода / Code structure

```text
start_forge.py             # Точка входа: перезапуск в venv, CLI / Entry point: venv re-exec, CLI
transcribe_input_audio.py  # Только транскрибация / Transcription only
rerender_videos.py         # Перерендер по moments.json / Re-render from moments.json
podcast_reels_forge/
├── pipeline.py            # Оркестрация стадий, очередь, отпечатки / Stage orchestration, queue, fingerprints
├── preflight.py           # Проверка окружения до старта / Environment check before the run
├── autonomy.py            # Лог-файл, отчёт, уведомления / Log file, report, notifications
├── run_report.py          # Отчёт о прогоне (_runs/*.json) / Run report
├── config.py              # Хелперы конфига / Config helpers
├── stages/                # Реализации стадий / Stage implementations
│   ├── fetch_stage.py         # yt-dlp
│   ├── transcribe_stage.py    # faster-whisper
│   ├── proofread_stage.py     # Вычитка LLM / LLM proofreading
│   ├── article_stage.py       # Лонгрид / Long-read
│   ├── analyze_stage.py       # scout → cleanup → judge → отбор / selection
│   └── video_stage.py         # Конфиги нарезки / Cut configs
├── analysis/              # Детерминированная часть анализа / Deterministic analysis
│   ├── chunking.py            # Чанки транскрипта / Transcript chunks
│   ├── validation.py          # Сверка цитат / Quote verification
│   ├── decisions.py           # keep/drop/merge по candidate_id
│   ├── ranking.py, scoring.py # MMR, квоты, веса / MMR, quotas, weights
│   ├── audio_features.py      # Громкость, паузы, темп / Loudness, pauses, rate
│   ├── speaker_turns.py       # Реплики по диаризации / Turns from diarization
│   └── term_check.py          # Перепроверка редких имён / Rare-name check
├── llm/
│   ├── providers.py       # llama.cpp, OpenAI, Anthropic, Gemini
│   └── schemas.py         # JSON-схемы для грамматики / JSON schemas for the grammar
├── sources/
│   ├── youtube.py         # Разбор ссылок, Data API / Link parsing, Data API
│   └── episode_metadata.py    # .info.json от yt-dlp
├── scripts/               # CLI отдельных стадий / Per-stage CLIs (python -m ...)
│   ├── analyze.py, transcribe.py, diarize.py, video_processor.py
│   ├── fetch_youtube.py, host_memory.py, rerender_videos.py
│   └── evaluate_prompts.py    # A/B промптов, golden set / Prompt A/B, golden set
└── utils/
    ├── face_track.py      # Кадр следит за говорящим / Speaker-following frame
    ├── face_crop.py       # YuNet в torch / YuNet in torch
    ├── active_speaker.py  # Light-ASD
    ├── burned_subtitles.py, subtitle_layout.py, subtitle_presets.py
    ├── subtitle_sync.py   # Перепроверка таймингов Whisper / Whisper timing re-check
    ├── word_alignment.py  # Тайминги для исправленного текста / Timings for corrected text
    ├── clip_intervals.py  # Интервал клипа с отступами / Padded clip interval
    ├── media_qa.py        # ffprobe-проверка клипа / Clip QA
    ├── fingerprint.py     # Отпечатки входов стадий / Stage input fingerprints
    ├── config_loader.py   # config.yaml + extends + config.local.yaml
    ├── llama_cpp_service.py, host_memory.py, run_lock.py
    └── ffmpeg.py, env.py, json_utils.py, logging_utils.py, reel_markdown.py
gui/                       # Статический GUI без сервера / Static, serverless GUI
prompts/{ru,en}/           # Шаблоны промптов / Prompt templates
```

RU: `sources/` отвечает на вопрос «какие ролики имеются в виду» и ничего не
скачивает; скачивание живёт в `stages/fetch_stage.py`. Разделение позволяет
тестировать разбор ссылок и работу с Data API без сети и без yt-dlp.

EN: `sources/` answers "which videos are meant" and downloads nothing; the
downloading lives in `stages/fetch_stage.py`. The split lets link parsing and the
Data API work be tested without a network and without yt-dlp installed.

## Проверки / Checks

Те же три проверки, что в CI / The same three checks CI runs:

```bash
pytest -q
ruff check .
mypy podcast_reels_forge --ignore-missing-imports
```

```bash
pytest tests/test_subtitle_layout.py -v   # один файл / one file
pytest -k preset                           # по имени / by name
ruff check . --fix                         # автоисправление / autofix
```

RU: версия ruff в CI закреплена (`.github/workflows/tests.yml`): новая версия
расширила набор правил по умолчанию и уронила CI без единого изменения кода.
Повышайте её только вместе с правками.

EN: CI pins ruff (`.github/workflows/tests.yml`): an unpinned release once
widened the default rule set and turned CI red with no code change. Bump it
deliberately, together with any fixes.

## Оценка качества отбора / Evaluating moment selection

RU: Эвристические метрики (`avg_score`, длительности) показывают, что модель
что-то нашла, но не то, нашла ли она *правильные* моменты. Для этого нужна
ручная разметка эпизода — golden set.

EN: Heuristic metrics (`avg_score`, durations) show that the model found
*something*, not whether it found the *right* moments. That needs a
hand-labelled episode — a golden set.

### 1. Разметка / Labelling

RU: Прогоните эпизод один раз и откройте `scout_candidates.json` и `reels.md`
— по ним удобно выбирать. Создайте `golden/<эпизод>.json`:

EN: Run the episode once and use `scout_candidates.json` and `reels.md` to
pick from. Create `golden/<episode>.json`:

```json
{
  "episode": "POS-7 - HG",
  "moments": [
    {
      "start": 412.0,
      "end": 468.0,
      "label": "must",
      "topics": ["школьная программа"],
      "note": "Самая сильная история эпизода"
    },
    {"start": 1120.5, "end": 1165.0, "label": "good", "topics": ["еда"]}
  ]
}
```

RU: `label` — `must` (нельзя пропустить), `good` (хороший момент), `ok`
(допустимый). `recall_must` считается отдельно: пропуск `must` — это провал,
а не потеря процента.

EN: `label` is `must` (must not be missed), `good`, or `ok`. `recall_must` is
reported separately: missing a `must` is a failure, not a lost percentage.

Имя файла — по эпизоду, без суффикса `.proofread`.

### 2. Сравнение вариантов / Comparing variants

```bash
python -m podcast_reels_forge.scripts.evaluate_prompts \
    --transcript "output/<эпизод>/<эпизод>.proofread.json" \
    --variants default,a,b
```

RU: Golden-файл подхватывается автоматически из `golden/<эпизод>.json`
(или укажите `--golden PATH`). Отчёт `prompt_eval.json` получит
`recall_must`, `recall_all` и `precision`, а выбор лучшего варианта начнёт
опираться на них вместо эвристики.

EN: The golden file is picked up automatically from `golden/<episode>.json`
(or pass `--golden PATH`). The `prompt_eval.json` report then carries
`recall_must`, `recall_all` and `precision`, and the best-variant pick is
based on those instead of the heuristic.

## Архитектура / Architecture

### Конвейер / Pipeline

RU: `run_pipeline()` в `pipeline.py` строит очередь эпизодов из `input/` (и
скачанного `fetch`), затем при `autonomy.scheduling: stage` прогоняет её стадия
за стадией: все транскрибации с одной загрузкой Whisper → все LLM-стадии
(proofread, article, analyze) в одной сессии llama-server → вся нарезка. Каждая
стадия каждого эпизода обёрнута в защиту от сбоя и пишет результат в отчёт
прогона. Перед стадией сравнивается отпечаток её входов (`.forge_state.json`):
совпал — стадия пропускается.

EN: `run_pipeline()` in `pipeline.py` builds the episode queue from `input/`
(plus what `fetch` downloaded) and, with `autonomy.scheduling: stage`, runs it
stage by stage: every transcription with one Whisper load → every LLM stage
(proofread, article, analyze) in one llama-server session → all the cutting.
Each stage of each episode runs inside a failure guard and lands in the run
report. A stage is skipped when its input fingerprint (`.forge_state.json`)
matches.

### Анализ / Analysis

RU: «LLM находит → Python доказывает → детерминированный отбор выбирает → LLM
пишет метаданные». Модель не может сдвинуть клип или переписать цитату:
cleanup и judge отвечают решениями по `candidate_id`. Подробно — в
[ANALYSIS_REPORT.md](ANALYSIS_REPORT.md) и [PROMPTS.md](PROMPTS.md).

EN: "LLM discovers → Python proves → a deterministic selector chooses → LLM
writes metadata". The model cannot move a clip or rewrite a quote: cleanup and
judge answer with decisions by `candidate_id`. See
[ANALYSIS_REPORT.md](ANALYSIS_REPORT.md) and [PROMPTS.md](PROMPTS.md).

### Пресеты субтитров / Subtitle presets

RU: источник правды — `utils/subtitle_presets.py`; копия для GUI
`gui/assets/subtitle-presets.js` генерируется командой
`python3 -m podcast_reels_forge.utils.subtitle_presets`. Меняете пресет —
перегенерируйте и закоммитьте оба файла.

EN: the source of truth is `utils/subtitle_presets.py`; the GUI copy
`gui/assets/subtitle-presets.js` is generated by
`python3 -m podcast_reels_forge.utils.subtitle_presets`. Change a preset —
regenerate and commit both files.

### Добавление новой модели / Adding a new model

1. Подготовьте GGUF-модель для llama.cpp (`llama_cpp.service.model_path`, лучше в `config.local.yaml`).
2. Назначьте её ролям в `llama_cpp.roles`.

## Соглашения / Conventions

### Именование

Naming

- Модули: `snake_case.py`
- Классы: `PascalCase`
- Функции/методы: `snake_case`
- Константы: `UPPER_SNAKE_CASE`
- Приватные: `_leading_underscore`

### Типизация

Все публичные функции должны иметь type hints:

All public functions should have type hints:

```python
def find_moments(
    provider: LLMProvider,
    segments: list[dict[str, Any]],
    duration: float,
    *,
    r_min: int = 30,
    r_max: int = 60,
) -> list[Moment]:
    """RU: Находит виральные моменты в транскрипции.

    EN: Finds viral moments in a transcript.
    """
    ...
```

### Docstrings

Используйте Google-стиль:

Use Google-style docstrings:

```python
def function(arg1: str, arg2: int) -> bool:
    """RU: Краткое описание функции.

    Более подробное описание, если необходимо.

    EN: Short function description.

    More detailed description, if needed.

    Args:
        arg1: Описание первого аргумента. / Description of the first argument.
        arg2: Описание второго аргумента. / Description of the second argument.

    Returns:
        Описание возвращаемого значения. / Description of the return value.

    Raises:
        ValueError: Когда arg2 отрицательный. / When arg2 is negative.
    """
```

## CI/CD

- `.github/workflows/tests.yml` — на каждый push в `main` и каждый PR: `ruff`
  (Python 3.10) и `pytest` на 3.10 и 3.12, `mypy` на 3.10.
- `.github/workflows/release.yml` — на тег `vX.Y.Z`: проверяет, что тег совпадает
  с `__version__`, берёт раздел версии из `CHANGELOG.md` и публикует GitHub Release.
- Dependabot обновляет GitHub Actions и pip-зависимости.

EN: `tests.yml` runs ruff, pytest (3.10, 3.12) and mypy on every push to `main`
and every PR. `release.yml` fires on a `vX.Y.Z` tag, checks the tag against
`__version__`, and publishes a GitHub Release with that version's CHANGELOG
section as the notes.

## Релизы / Releases

1. Перенесите `## [Unreleased]` в `## [X.Y.Z] — ГГГГ-ММ-ДД` в `CHANGELOG.md` и
   добавьте ссылку сравнения внизу файла.
2. Поднимите `__version__` в `podcast_reels_forge/__init__.py`.
3. Закоммитьте, затем поставьте и отправьте тег — релиз опубликует CI:

   ```bash
   git commit -am "chore(release): vX.Y.Z"
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin main vX.Y.Z
   ```

EN: move `## [Unreleased]` to `## [X.Y.Z] — YYYY-MM-DD` (plus the compare link
at the bottom), bump `__version__`, commit, then push an annotated `vX.Y.Z`
tag — the release workflow publishes the GitHub Release.

Версионирование — [SemVer](https://semver.org/): новые возможности — minor,
исправления — patch. / Versioning follows SemVer.

## Troubleshooting / Устранение неполадок

- **ImportError в тестах / ImportError in tests**: `pip install -e .`
- **FFmpeg не найден / FFmpeg not found**: `ffmpeg -version`; путь можно задать
  через `FORGE_FFMPEG`. / Set the path via `FORGE_FFMPEG`.
- **GUI не видит новый ключ / GUI misses a new key**: каждый ключ конфига должен
  быть в шаблоне GUI (`gui/assets/app.js`), иначе экспорт из GUI его потеряет;
  строки интерфейса — в словарях `ru` и `en`. / Every config key must be in the
  GUI template, or a GUI export drops it; UI strings live in both `ru` and `en`.
