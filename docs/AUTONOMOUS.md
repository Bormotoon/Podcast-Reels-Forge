# Запуск без присмотра / Unattended runs

RU: Как запускать Forge по расписанию (например, каждую ночь по каналу) так,
чтобы прогон не падал из-за одного эпизода, не тратил время на повторную
работу и сообщал о результате. Разбор, из которого выросли эти возможности, —
[AUTONOMY_REVIEW.md](AUTONOMY_REVIEW.md).

EN: How to run Forge on a schedule (say, nightly over a channel) so a run does
not die on one episode, does not redo work, and reports its outcome. The
review these features came from is [AUTONOMY_REVIEW.md](AUTONOMY_REVIEW.md).

## Расписание / Scheduling

### systemd (рекомендуется)

`~/.config/systemd/user/forge.service`:

```ini
[Unit]
Description=Podcast Reels Forge nightly run

[Service]
Type=oneshot
WorkingDirectory=%h/Podcast-Reels-Forge
ExecStart=%h/Podcast-Reels-Forge/whisper-env/bin/python start_forge.py --no-progress --quiet
# 3 = частичный успех: для systemd это не сбой юнита.
SuccessExitStatus=3 75
TimeoutStartSec=12h
```

`~/.config/systemd/user/forge.timer`:

```ini
[Unit]
Description=Nightly Podcast Reels Forge

[Timer]
OnCalendar=*-*-* 02:30
Persistent=true

[Install]
WantedBy=timers.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now forge.timer
loginctl enable-linger "$USER"   # запускать и без открытой сессии
```

### cron

```cron
30 2 * * * cd $HOME/Podcast-Reels-Forge && ./whisper-env/bin/python start_forge.py --no-progress --quiet
```

Каналы и фильтры задаются в блоке `youtube:` конфига (`sources`, `exclude`,
`limit`, `since`), так что команда запуска не меняется.

## Коды возврата / Exit codes

| Код | Значение |
|---|---|
| 0 | всё прошло |
| 3 | частично: хотя бы одна стадия эпизода упала или YouTube был недоступен; остальные эпизоды обработаны |
| 1 | фатально: не прошла предполётная проверка или прогон прервался целиком |
| 75 | уже идёт другой прогон (блокировка), этот пропущен |

## Отчёт прогона / Run report

Каждый прогон пишет `<paths.output_dir>/_runs/<время>.json` и обновляет
`latest.json` (папка настраивается в `autonomy.runs_dir`):

```json
{
  "outcome": "partial",
  "exit_code": 3,
  "episodes_total": 4,
  "episodes_failed": ["2026-09-20 - Show [abc123def45]"],
  "clips_total": 41,
  "events": [{"level": "error", "message": "YouTube недоступен (...); работаю с уже скачанным"}],
  "episodes": {
    "2026-09-20 - Show [abc123def45]": {
      "status": "failed",
      "stages": {
        "audio": {"status": "cached", "seconds": 0.0},
        "transcribe": {"status": "failed", "seconds": 61.2, "detail": "RuntimeError: CUDA failed: out of memory"}
      }
    }
  }
}
```

Статусы стадий: `done`, `cached` (результат уже был и входы не менялись),
`skipped`, `failed`.

## Логи / Logs

`logs/forge.log` с ротацией в полночь, хранится `autonomy.log_keep_days` дней.
В файл всегда пишется уровень INFO, даже при `--quiet`. Вывод llama-server —
отдельно, в `llama_cpp.service.log_file`.

## Уведомления / Notifications

```yaml
autonomy:
  notify:
    when: failure        # always | failure | never
    command: 'notify-send "Forge" "$FORGE_SUMMARY"'
    webhook: "https://hooks.slack.com/services/..."
```

Команда получает переменные окружения `FORGE_OUTCOME`, `FORGE_EXIT_CODE`,
`FORGE_SUMMARY` и `FORGE_REPORT` (путь к отчёту). Вебхук получает POST с JSON:
`text` (совместимо со Slack/Discord/Mattermost), `outcome`,
`episodes_failed`, `clips_total`, `fatal_error`, `report`. Пример отправки в
Telegram через команду:

```yaml
    command: >-
      curl -s "https://api.telegram.org/bot$TG_TOKEN/sendMessage"
      -d chat_id="$TG_CHAT" --data-urlencode text="Forge: $FORGE_SUMMARY"
```

## Что защищает прогон / What keeps a run alive

- **Изоляция эпизодов.** Ошибка любой стадии записывается в отчёт и стоит
  только этого эпизода (для необязательных стадий — только этой стадии).
- **Блокировка** `autonomy.lock_file`: второй запуск выходит с кодом 75, а не
  убивает llama-server первого.
- **Предполётная проверка**: ffmpeg, faster-whisper, llama-server и модель
  (если сервер ещё не отвечает), pyannote и `PYANNOTE_TOKEN` при диаризации,
  yt-dlp при загрузке, свободное место (`autonomy.min_free_disk_gb`). Ошибка —
  прогон не начинается (код 1). `--skip-preflight` отключает проверку.
- **Недоступный YouTube** не обрывает прогон: обрабатывается уже скачанное.
- **Прерванные загрузки** (фрагменты `.fNNN` без склейки) не принимаются за
  эпизоды; yt-dlp докачивает их в следующий раз.
- **Нехватка видеопамяти у Whisper**: batch уменьшается вдвое, затем CPU.
- **yt-dlp обновляется сам** раз в `youtube.self_update_days` дней
  (`youtube.self_update`).

## Повторная работа / Redoing work

- **Отпечатки стадий** (`.forge_state.json` в папке эпизода): анализ
  пересчитывается, когда меняются транскрипт, `processing`, промпты, роли или
  метаданные эпизода; нарезка — когда меняются моменты, транскрипт субтитров,
  настройки видео, субтитров, экспорта или фильтров качества (старые ролики
  при этом удаляются). Результаты, сделанные до появления отпечатков,
  принимаются как есть.
- **Кэш ответов LLM** (`processing.analysis.llm_cache`): анализ, упавший на
  середине, при перезапуске не повторяет уже полученные ответы.
- Анализ, который честно не нашёл ничего стоящего, помечается завершённым и
  не пересчитывается каждую ночь.

## Скорость / Throughput

| Настройка | Что даёт |
|---|---|
| `autonomy.scheduling: stage` | Whisper загружается один раз на очередь, llama-server стартует один раз на все LLM-стадии |
| `proofread.scope: clips` | вычитываются только отрезки выбранных клипов (если статья выключена) |
| вычитка возвращает только исправленные сегменты | кратно меньше генерации на самой дорогой стадии |
| `role_overrides.*.n_predict` | меньший потолок генерации для scout/cleanup/judge |
| `audio.listening_copy: false` | не создаётся MP3 320k, которым модели не пользуются |
| `audio.delete_wav_after_analysis: true` | WAV 16 кГц (~115 МБ/час) удаляется после анализа |
| `quality_filters.render_rejected: false` | отбракованные клипы не кодируются, только перечисляются в `reels/rejected.json` |
| ffmpeg с NVENC и libass | субтитры вжигаются на GPU (иначе — libx264) |
