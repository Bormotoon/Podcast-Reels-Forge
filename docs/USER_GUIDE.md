# User Guide / Руководство пользователя

## RU: Что это

Podcast Reels Forge автоматически находит сильные моменты в длинных подкастах и режет их в вертикальные клипы для Reels / Shorts / TikTok.

Пайплайн по умолчанию:

1. Транскрибация через `faster-whisper`.
2. Опциональная диаризация через `pyannote`.
3. Staged analysis через локальный Ollama-only Gemma lineup.
4. Нарезка видео через FFmpeg, face-aware crop и burned subtitles через `pycaps`.

Важно:

- Дефолтный workflow локальный и Gemma-only.
- Анализ пишет финальные артефакты в `output/<file_stem>/gemma4_26b/`.
- Транскрипт теперь содержит `segments[].words`, `sentences`, `language_confidence`, `source_audio` и `timing_version`.
- Если рядом с видео есть одноимённый `mp3`, Forge использует его. Иначе аудио извлекается автоматически.

---

## RU: Быстрый старт

### 1) Требования

- Python 3.10+
- FFmpeg в `PATH`
- Ollama
- NVIDIA GPU желательно, но не обязательно

### 2) Установка

```bash
python3 -m venv whisper-env
source whisper-env/bin/activate
pip install -r requirements.txt
playwright install chromium
```

### 3) Входные файлы

Положите видео в `input/` и запустите:

```bash
python3 start_forge.py
```

Полезные флаги:

- `--verbose` - подробные логи
- `--quiet` - только ошибки
- `--no-skip-existing` - пересчитать все стадии
- `--autotune` - более безопасные параметры для текущего железа

---

## RU: Где лежат результаты

Финальные результаты одной обработки:

- `output/<file_stem>/audio.json` - транскрипт
- `output/<file_stem>/diarization.json` - speaker turns, если diarization включена
- `output/<file_stem>/gemma4_26b/analysis_manifest.json`
- `output/<file_stem>/gemma4_26b/scout_candidates.json`
- `output/<file_stem>/gemma4_26b/cleaned_candidates.json`
- `output/<file_stem>/gemma4_26b/refined_candidates.json`
- `output/<file_stem>/gemma4_26b/judge_report.json`
- `output/<file_stem>/gemma4_26b/moments.json`
- `output/<file_stem>/gemma4_26b/reels.md`
- `output/<file_stem>/gemma4_26b/reels/` - клипы, `reel_XX.md`, `reel_XX.srt`
- `output/<file_stem>/gemma4_26b/reels_preview.mp4`

`gemma4_26b` - это имя финальной judge-model папки по умолчанию.

---

## RU: Перерендер из готового moments.json

Если нужно поменять crop / codec / subtitles, можно перерендерить клипы без нового анализа:

```bash
python3 rerender_videos.py --model gemma4_26b --smart-crop-face --replace
```

Если хотите просто посмотреть доступные опции:

```bash
python3 rerender_videos.py --help
```

---

## RU: Smart crop

`smart_crop_face: true` включает offline face-aware framing:

1. Берется несколько кадров-семплов.
2. OpenCV Haar cascades ищут лицо.
3. Если лицо найдено, crop двигается плавно и стабильно.
4. Если лицо не найдено, используется center crop с явным fallback log.

---

## RU: Настройка через config.yaml

Ключевые блоки:

- `transcription` - качество транскрипции
- `ollama.roles` - staged Gemma mapping
- `ollama.role_overrides` - таймауты и chunk-size по ролям
- `processing` - квоты клипов и quality filters
- `video` - crop, bitrate, NVENC
- `subtitles` - burned subtitles через `pycaps`
- `diarization` - speaker turns через pyannote

Если модель Ollama долго думает, правьте `ollama.watchdog` или `ollama.role_overrides`.

---

## RU: Частые проблемы

- OOM на транскрипции: уменьшите `transcription.model` или переключитесь на `cpu`.
- Ollama не отвечает: проверьте `ollama` и увеличьте `ollama.watchdog.first_token_timeout`.
- Сабы не вшиваются: выполните `playwright install chromium`.
- Crop кажется неудачным: отключите `video.smart_crop_face` или увеличьте `video.face_samples`.

---

## EN: What it is

Podcast Reels Forge finds strong moments in long-form podcasts and cuts them into vertical clips for Reels / Shorts / TikTok.

Default pipeline:

1. Transcription via `faster-whisper`.
2. Optional speaker diarization via `pyannote`.
3. Staged analysis through the local Ollama-only Gemma lineup.
4. FFmpeg rendering with face-aware crop and burned subtitles via `pycaps`.

Key points:

- The default workflow is local-first and Gemma-only.
- Final artifacts are written to `output/<file_stem>/gemma4_26b/`.
- The transcript includes additive timing metadata such as `words`, `sentences`, `language_confidence`, `source_audio`, and `timing_version`.

---

## EN: Output layout

- `output/<file_stem>/audio.json`
- `output/<file_stem>/diarization.json`
- `output/<file_stem>/gemma4_26b/analysis_manifest.json`
- `output/<file_stem>/gemma4_26b/scout_candidates.json`
- `output/<file_stem>/gemma4_26b/cleaned_candidates.json`
- `output/<file_stem>/gemma4_26b/refined_candidates.json`
- `output/<file_stem>/gemma4_26b/judge_report.json`
- `output/<file_stem>/gemma4_26b/moments.json`
- `output/<file_stem>/gemma4_26b/reels.md`
- `output/<file_stem>/gemma4_26b/reels/`
- `output/<file_stem>/gemma4_26b/reels_preview.mp4`

---

## EN: Re-render existing moments

```bash
python3 rerender_videos.py --model gemma4_26b --smart-crop-face --replace
```

---

## EN: Troubleshooting

- Transcription OOM: lower `transcription.model` or switch to `cpu`.
- Ollama stalls: raise `ollama.watchdog.first_token_timeout`.
- Subtitles fail: run `playwright install chromium`.
- Crop feels off: disable `video.smart_crop_face` or increase `video.face_samples`.
