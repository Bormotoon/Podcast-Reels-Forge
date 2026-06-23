# Contributing / Как внести вклад

Thanks for your interest in improving **Podcast Reels Forge**! / Спасибо за интерес к проекту!

This document is bilingual: English first, then Russian. / Документ двуязычный: сначала английский, затем русский.

---

## English

### Getting started

```bash
git clone https://github.com/Bormotoon/Podcast-Reels-Forge.git
cd Podcast-Reels-Forge

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

You also need **ffmpeg** on your `PATH` (or point `FORGE_FFMPEG` at it). The
default analysis stage talks to a local **llama.cpp** server — see the
[README](README.en.md) for the full setup.

### Before you open a pull request

Run the same checks CI runs, locally:

```bash
pytest -q
ruff check .
mypy podcast_reels_forge --ignore-missing-imports
```

All three must be green. New behaviour should come with tests in `tests/`.

### Pull request guidelines

- Branch off `main`; keep PRs focused on a single change.
- Write [Conventional Commits](https://www.conventionalcommits.org/) messages
  (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`…).
- Update the docs (`README.md`, `README.en.md`, `docs/`) when behaviour changes.
- Keep the GUI bilingual: every user-facing string lives in the `ru` **and** `en`
  dictionaries in `gui/assets/app.js`.
- Describe what changed and why, and how you verified it.

### Reporting bugs / requesting features

Use the issue templates. Include your OS, Python version, GPU (if relevant), the
command you ran, and the full error output.

---

## Русский

### Начало работы

```bash
git clone https://github.com/Bormotoon/Podcast-Reels-Forge.git
cd Podcast-Reels-Forge

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Также нужен **ffmpeg** в `PATH` (или укажите путь через `FORGE_FFMPEG`). Стадия
анализа по умолчанию обращается к локальному серверу **llama.cpp** — полная
настройка описана в [README](README.md).

### Перед открытием pull request

Запустите локально те же проверки, что и CI:

```bash
pytest -q
ruff check .
mypy podcast_reels_forge --ignore-missing-imports
```

Все три должны быть зелёными. Новое поведение сопровождайте тестами в `tests/`.

### Правила для pull request

- Ветвитесь от `main`; один PR — одно логическое изменение.
- Используйте [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`…).
- Обновляйте документацию (`README.md`, `README.en.md`, `docs/`), если меняется
  поведение.
- Держите GUI двуязычным: каждая видимая пользователю строка должна быть в
  словарях `ru` **и** `en` в `gui/assets/app.js`.
- Опишите, что изменилось, зачем и как вы это проверили.

### Баги и запросы фич

Используйте шаблоны issue. Укажите ОС, версию Python, GPU (если важно),
запущенную команду и полный вывод ошибки.
