# Security Policy / Политика безопасности

## Supported versions / Поддерживаемые версии

Only the latest release gets security fixes. / Исправления безопасности выходят только для последнего релиза.

## Reporting a vulnerability (English)

Please **do not** open a public issue for security vulnerabilities.

Instead, report them privately via GitHub's
[private vulnerability reporting](https://github.com/Bormotoon/Podcast-Reels-Forge/security/advisories/new)
(Security → Advisories → Report a vulnerability), or by contacting the repository
owner directly.

Include a description of the issue, steps to reproduce, and the affected version
or commit. We aim to acknowledge reports within a few days.

### Notes

- This is a local, command-line tool. Transcription, analysis and rendering run
  on your machine against your own local llama.cpp server. The network is used
  only for:
  - downloading model weights on first use (Whisper, YuNet, Light-ASD, pyannote);
  - the YouTube `fetch` stage and its weekly `yt-dlp` self-update
    (`youtube.self_update`), when you use YouTube;
  - the rare-name spell check (`proofread.terms`, off by default) — only the
    suspect word and one neighbour are sent to Wiktionary;
  - cloud LLM providers and the notification webhook, only if you configure them.
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

- Это локальный CLI-инструмент. Транскрибация, анализ и рендер идут на вашей
  машине через ваш локальный llama.cpp-сервер. В сеть обращаются только:
  - загрузка весов моделей при первом запуске (Whisper, YuNet, Light-ASD, pyannote);
  - стадия `fetch` с YouTube и еженедельное самообновление `yt-dlp`
    (`youtube.self_update`) — если вы работаете с YouTube;
  - перепроверка редких имён (`proofread.terms`, по умолчанию выключена) — в
    Викисловарь уходит только подозрительное слово и одно соседнее;
  - облачные LLM-провайдеры и вебхук уведомлений — только если вы их настроили.
- API-ключи и токены хранятся в git-игнорируемом `.env` (см. `.env.example`).
  Никогда не коммитьте секреты. Если ключ всё же утёк — немедленно отзовите его.
