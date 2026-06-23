# Security Policy / Политика безопасности

## Reporting a vulnerability (English)

Please **do not** open a public issue for security vulnerabilities.

Instead, report them privately via GitHub's
[private vulnerability reporting](https://github.com/Bormotoon/Podcast-Reels-Forge/security/advisories/new)
(Security → Advisories → Report a vulnerability), or by contacting the repository
owner directly.

Include a description of the issue, steps to reproduce, and the affected version
or commit. We aim to acknowledge reports within a few days.

### Notes

- This is a local, command-line tool. The default pipeline runs entirely on your
  machine and makes no outbound network calls beyond your own local llama.cpp
  server.
- API keys and tokens live in a git-ignored `.env` file (see `.env.example`).
  Never commit secrets. If a key is ever exposed, rotate it immediately.

## Сообщение об уязвимости (Русский)

Пожалуйста, **не** открывайте публичный issue для уязвимостей.

Сообщайте о них приватно через
[private vulnerability reporting](https://github.com/Bormotoon/Podcast-Reels-Forge/security/advisories/new)
GitHub (Security → Advisories → Report a vulnerability) или напрямую владельцу
репозитория.

Укажите описание проблемы, шаги воспроизведения и затронутую версию/коммит. Мы
постараемся отреагировать в течение нескольких дней.

### Примечания

- Это локальный CLI-инструмент. Пайплайн по умолчанию работает полностью на вашей
  машине и не делает внешних сетевых запросов, кроме вашего локального
  llama.cpp-сервера.
- API-ключи и токены хранятся в git-игнорируемом `.env` (см. `.env.example`).
  Никогда не коммитьте секреты. Если ключ всё же утёк — немедленно отзовите его.
