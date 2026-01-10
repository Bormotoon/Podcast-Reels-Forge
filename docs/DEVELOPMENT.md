# Руководство для разработчиков / Developer Guide

## Установка для разработки / Development setup

```bash
# Клонировать репозиторий / Clone the repository
git clone https://github.com/yourusername/podcast-reels-forge.git
cd podcast-reels-forge

# Создать виртуальное окружение / Create a virtual environment
python3 -m venv whisper-env
source whisper-env/bin/activate

# Установить зависимости / Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Структура кода / Code structure

```text
podcast_reels_forge/
├── __init__.py          # Версия пакета / Package version
├── __main__.py          # python -m podcast_reels_forge
├── cli.py               # CLI интерфейс пакета / Package CLI
├── pipeline.py          # Оркестрация пайплайна / Pipeline orchestration
├── scripts/             # CLI скрипты этапов / Stage CLI scripts
│   ├── __init__.py
│   ├── analyze.py       # Анализ с LLM / LLM analysis
│   ├── transcribe.py    # Транскрипция (обёртка) / Transcription wrapper
│   ├── video_processor.py  # Обработка видео / Video processing
│   ├── diarize.py       # Диаризация / Diarization
│   └── evaluate_prompts.py # A/B тестирование / A/B prompt evaluation
├── stages/              # Реализации этапов / Stage implementations
│   ├── __init__.py
│   ├── transcribe_stage.py  # Логика транскрипции / Transcription logic
│   ├── analyze_stage.py     # Метрики анализа / Analysis metrics
│   └── video_stage.py       # Конфиги видео / Video configs
├── llm/                 # LLM провайдеры / LLM providers
│   ├── __init__.py
│   └── providers.py     # Ollama, OpenAI, Anthropic, Gemini
└── utils/               # Утилиты / Utilities
    ├── __init__.py
    ├── json_utils.py    # Извлечение JSON из текста / Extract JSON from text
    └── logging_utils.py # Настройка логирования / Logging setup
```

## Запуск тестов / Running tests

```bash
# Все тесты / All tests
pytest

# С покрытием / With coverage
pytest --cov=podcast_reels_forge

# Конкретный файл / Specific file
pytest tests/test_pipeline.py -v

# Только быстрые тесты / Only fast tests
pytest -m "not slow"
```

## Проверка типов / Type checking

```bash
# Проверка всего пакета / Check the whole package
mypy podcast_reels_forge/

# Строгий режим / Strict mode
mypy podcast_reels_forge/ --strict
```

## Форматирование кода / Code formatting

```bash
# Black
black podcast_reels_forge/ tests/

# isort
isort podcast_reels_forge/ tests/

# Проверка без изменений / Check without changing files
black --check podcast_reels_forge/
isort --check podcast_reels_forge/
```

## Архитектура / Architecture

### Pipeline (pipeline.py)

Основной оркестратор, который:

1. Читает конфигурацию
1. Находит входные файлы
1. Последовательно запускает этапы
1. Передаёт данные между этапами

Main orchestrator that:

1. Reads configuration
1. Discovers input files
1. Runs stages sequentially
1. Passes data between stages

```python
def run_pipeline(*, conf: dict, repo_dir: Path, quiet: bool, verbose: bool) -> None:
    # 1. Transcribe
    run_module("podcast_reels_forge.scripts.transcribe", args, ...)
    
    # 2. Diarize (optional)
    if diar_enabled:
        run_module("podcast_reels_forge.scripts.diarize", args, ...)
    
    # 3. Analyze
    run_module("podcast_reels_forge.scripts.analyze", args, ...)
    
    # 4. Cut video
    run_module("podcast_reels_forge.scripts.video_processor", args, ...)
```

### LLM Providers (llm/providers.py)

Абстракция для работы с разными LLM:

Abstraction layer for multiple LLM backends:

```python
class LLMProvider(Protocol):
    def generate(self, prompt: str, *, temperature: float, timeout: int) -> str:
        ...

class OllamaProvider:
    def __init__(self, cfg: OllamaConfig): ...
    def generate(self, prompt: str, *, temperature: float, timeout: int) -> str: ...

class OpenAIProvider:
    def __init__(self, cfg: OpenAIConfig): ...
    def generate(self, prompt: str, *, temperature: float, timeout: int) -> str: ...
```

### Добавление нового провайдера

Adding a new provider

1. Создайте dataclass для конфигурации:

   ```python
   @dataclass(frozen=True)
   class NewProviderConfig:
       api_key: str
       model: str
   ```

1. Реализуйте класс провайдера:

   ```python
   class NewProvider:
       def __init__(self, cfg: NewProviderConfig):
           self.cfg = cfg

       def generate(self, prompt: str, *, temperature: float, timeout: int) -> str:
           # Реализация API вызова / API call implementation
           ...
   ```

1. Добавьте в `create_provider()` в `scripts/analyze.py`

1. Обновите pipeline.py для поддержки нового провайдера

### JSON Utils (utils/json_utils.py)

Утилиты для работы с JSON от LLM:

Utilities for parsing JSON from LLM output:

```python
def extract_first_json_object(text: str) -> dict:
    """RU: Извлекает первый JSON объект из текста.

    Обрабатывает случаи:
    - JSON внутри markdown-блока
    - JSON с текстом до/после
    - Невалидный JSON (базовый ремонт)

    EN: Extracts the first JSON object from text.

    Handles:
    - JSON inside a markdown block
    - JSON with surrounding text
    - Invalid JSON (basic repair)
    """
```

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
    segments: List[Dict[str, Any]],
    duration: float,
    *,
    r_min: int = 30,
    r_max: int = 60,
) -> List[Moment]:
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

### GitHub Actions

Workflow `.github/workflows/ci.yml`:

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: pytest
      - run: mypy podcast_reels_forge/
```

## Релизы / Releases

1. Обновите версию в `podcast_reels_forge/__init__.py`
1. Обновите CHANGELOG.md
1. Создайте тег:

   ```bash
   git tag -a v0.2.0 -m "Release v0.2.0"
   git push origin v0.2.0
   ```

- Bump the version in `podcast_reels_forge/__init__.py`
- Update CHANGELOG.md
- Create a tag:

    ```bash
    git tag -a v0.2.0 -m "Release v0.2.0"
    git push origin v0.2.0
    ```

## Troubleshooting / Устранение неполадок

### Тесты падают с ImportError

Tests fail with ImportError

```bash
# Убедитесь, что пакет установлен / Ensure the package is installed
pip install -e .
```

### mypy не находит модули

mypy can't find modules

```bash
# Проверьте mypy.ini
[mypy]
python_version = 3.10
warn_return_any = True
warn_unused_configs = True
ignore_missing_imports = True
```

### FFmpeg не работает

FFmpeg is not working

```bash
# Проверьте установку / Check installation
ffmpeg -version

# На Ubuntu
sudo apt install ffmpeg
```
