# User Guide / Руководство пользователя

## RU: Что это

Podcast Reels Forge автоматически находит сильные моменты в длинных подкастах и режет их в вертикальные клипы для Reels / Shorts / TikTok.

Пайплайн по умолчанию:

1. Транскрибация через `faster-whisper`.
2. Опциональная диаризация через `pyannote`.
3. Staged analysis через локальный llama.cpp-only Gemma 4 lineup.
4. Нарезка видео через FFmpeg, face-aware crop (MediaPipe) и вшитые ASS-субтитры через ffmpeg.

Важно:

- Дефолтный workflow локальный и Gemma-only.
- Анализ пишет финальные артефакты в `output/<file_stem>/gemma4/`.
- Транскрипт теперь содержит `segments[].words`, `sentences`, `language_confidence`, `source_audio` и `timing_version`.
- Если рядом с видео есть одноимённый `mp3`, Forge использует его. Иначе аудио извлекается автоматически.

---

## RU: Быстрый старт

### 1) Требования

- Python 3.10+
- FFmpeg в `PATH`
- llama.cpp (`llama-server`)
- NVIDIA GPU желательно, но не обязательно

### 2) Установка

```bash
python3 -m venv whisper-env
source whisper-env/bin/activate
pip install -r requirements.txt
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
- `output/<file_stem>/gemma4/analysis_manifest.json`
- `output/<file_stem>/gemma4/scout_candidates.json`
- `output/<file_stem>/gemma4/cleaned_candidates.json`
- `output/<file_stem>/gemma4/refined_candidates.json`
- `output/<file_stem>/gemma4/judge_report.json`
- `output/<file_stem>/gemma4/moments.json`
- `output/<file_stem>/gemma4/reels.md`
- `output/<file_stem>/gemma4/reels/` - клипы, `reel_XX.md`, `reel_XX.srt`
- `output/<file_stem>/gemma4/reels_preview.mp4`

`gemma4` - это имя финальной judge-model папки по умолчанию.

---

## RU: Перерендер из готового moments.json

Если нужно поменять crop / codec / subtitles, можно перерендерить клипы без нового анализа:

```bash
python3 rerender_videos.py --model gemma4 --smart-crop-face --replace
```

Если хотите просто посмотреть доступные опции:

```bash
python3 rerender_videos.py --help
```

---

## RU: Smart crop

`smart_crop_face: true` включает offline face-aware framing:

1. Берется несколько кадров-семплов.
2. MediaPipe Face Detection ищет лицо.
3. Если лицо найдено, crop двигается плавно и стабильно.
4. Если лицо не найдено, используется center crop с явным fallback log.

---

## RU: Настройка через config.yaml

Ключевые блоки:

- `transcription` - качество транскрипции
- `llama_cpp.roles` - staged Gemma mapping
- `llama_cpp.role_overrides` - таймауты и chunk-size по ролям
- `processing` - квоты клипов и quality filters
- `video` - crop, bitrate, NVENC
- `subtitles` - вшитые ASS-субтитры через ffmpeg
- `diarization` - speaker turns через pyannote

Если модель llama.cpp долго думает, правьте `llama_cpp.watchdog` или `llama_cpp.role_overrides`.

---

## RU: Частые проблемы

- OOM на транскрипции: уменьшите `transcription.model` или переключитесь на `cpu`.
- llama.cpp не отвечает: проверьте `llama_cpp` и увеличьте `llama_cpp.watchdog.first_token_timeout`.
- Crop кажется неудачным: отключите `video.smart_crop_face` или увеличьте `video.face_samples`.

---

## EN: What it is

Podcast Reels Forge finds strong moments in long-form podcasts and cuts them into vertical clips for Reels / Shorts / TikTok.

Default pipeline:

1. Transcription via `faster-whisper`.
2. Optional speaker diarization via `pyannote`.
3. Staged analysis through the local llama.cpp-only Gemma 4 lineup.
4. FFmpeg rendering with face-aware crop (MediaPipe) and burned-in ASS subtitles via ffmpeg.

Key points:

- The default workflow is local-first and Gemma-only.
- Final artifacts are written to `output/<file_stem>/gemma4/`.
- The transcript includes additive timing metadata such as `words`, `sentences`, `language_confidence`, `source_audio`, and `timing_version`.

---

## EN: Output layout

- `output/<file_stem>/audio.json`
- `output/<file_stem>/diarization.json`
- `output/<file_stem>/gemma4/analysis_manifest.json`
- `output/<file_stem>/gemma4/scout_candidates.json`
- `output/<file_stem>/gemma4/cleaned_candidates.json`
- `output/<file_stem>/gemma4/refined_candidates.json`
- `output/<file_stem>/gemma4/judge_report.json`
- `output/<file_stem>/gemma4/moments.json`
- `output/<file_stem>/gemma4/reels.md`
- `output/<file_stem>/gemma4/reels/`
- `output/<file_stem>/gemma4/reels_preview.mp4`

---

## EN: Re-render existing moments

```bash
python3 rerender_videos.py --model gemma4 --smart-crop-face --replace
```

---

## EN: Troubleshooting

- Transcription OOM: lower `transcription.model` or switch to `cpu`.
- llama.cpp stalls: raise `llama_cpp.watchdog.first_token_timeout`.
- Crop feels off: disable `video.smart_crop_face` or increase `video.face_samples`.
