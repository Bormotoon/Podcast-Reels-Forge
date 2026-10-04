# User Guide / Руководство пользователя

[Русский](#ru-что-это) · [English](#en-what-it-is)

## RU: Что это

Podcast Reels Forge находит сильные моменты в длинных подкастах и интервью и
режет их в вертикальные клипы для Reels / Shorts / TikTok. Всё работает
локально: распознавание, языковая модель, монтаж.

Конвейер по умолчанию:

0. **fetch** — (если задан YouTube) ролик, плейлист или канал скачиваются в `input/youtube/`.
1. **transcribe** — `faster-whisper large-v3` с пословными таймкодами.
2. **diarize** — (опционально) разделение по спикерам через `pyannote`.
3. **proofread** — gemma4 исправляет орфографию и пунктуацию.
4. **article** — лонгрид по эпизоду (`<имя>.article.md`).
5. **analyze** — поиск моментов: scout → cleanup → judge на локальной Gemma 4 через llama.cpp, каждая цитата сверяется с транскриптом.
6. **cut** — клипы 9:16: края по паузам в речи, кадр следит за говорящим, вшитые субтитры.

---

## RU: Быстрый старт

### 1) Требования

- Python 3.10+
- FFmpeg в `PATH` (лучше сборка с NVENC и libass)
- llama.cpp (`llama-server`) и GGUF-модель Gemma 4
- NVIDIA GPU с CUDA — фактически обязательна (Whisper, слежение за лицами, кодирование)

### 2) Установка

```bash
git clone https://github.com/Bormotoon/Podcast-Reels-Forge.git
cd Podcast-Reels-Forge
python3 -m venv whisper-env
source whisper-env/bin/activate
pip install -r requirements.txt
pip install -e ".[youtube,diarization]"   # опционально
cp .env.example .env                       # опционально: токены и ключи
```

Пути к модели llama.cpp и другие настройки своей машины положите в
`config.local.yaml` (он в `.gitignore` и подмешивается поверх `config.yaml`):

```yaml
llama_cpp:
  service:
    model_path: "/models/gemma-4-26b-q4.gguf"
```

### 3) Входные файлы

Положите видео (mp4, mkv, mov) или аудио в `input/` и запустите:

```bash
python3 start_forge.py
```

Папка просматривается вместе с подпапками. Перед стартом preflight проверит
ffmpeg, llama-server, токены и свободное место и скажет, чего не хватает.

Полезные флаги:

- `--verbose` / `--quiet` — подробные логи / только ошибки
- `--no-skip-existing` — пересчитать все стадии
- `--only analyze,cut` / `--skip cut` — запустить часть стадий
- `--list-stages` — показать стадии по порядку
- `--autotune` — более безопасные параметры для текущего железа

### 4) Или забрать материал с YouTube

Нужен `yt-dlp` (`pip install -U yt-dlp`); ключ `YOUTUBE_API_KEY` не обязателен.

```bash
# Один ролик через весь пайплайн
python3 start_forge.py --youtube "https://youtu.be/D6WjXRJt1DA" --yt-video

# Весь канал, всё кроме нарезки (по умолчанию качается только аудио)
python3 start_forge.py --youtube "@pedobraz" --skip cut

# Посмотреть отбор, ничего не скачивая
python3 start_forge.py --youtube "@pedobraz" --yt-limit 5 --yt-list
```

Файлы падают в `input/youtube/` под именем `ГГГГ-ММ-ДД - Заголовок [id]`, а прогон
сужается до названных роликов. Подробности — в [CONFIGURATION.md](CONFIGURATION.md#youtube--загрузка-с-youtube)
и в README.

### 5) Без присмотра

Ночной прогон по таймеру, отчёт о каждой стадии и уведомления описаны в
[AUTONOMOUS.md](AUTONOMOUS.md).

---

## RU: Где лежат результаты

```text
output/<имя>/
  <имя>.json, <имя>.srt                 транскрипт
  <имя>.proofread.json, .proofread.srt  вычитанный транскрипт
  <имя>.article.md, .article.json       лонгрид
  diarization.json                      спикеры (если диаризация включена)
  gemma4_26b/                           папка модели анализа
    moments.json                        финальный список клипов
    reels.md                            сводка клипов
    rejected_candidates.json            отброшенные кандидаты с причиной
    analysis_metrics.json               метрики прогона анализа
    reels/reel_XX.mp4                   клипы
    reels/reel_XX.md, reel_XX.srt       описание с хештегами и субтитры
    reels/framing/reel_XX.json          кого и когда показывал кадр
    reels/subtitle_sync.json            сдвиг таймингов после перепроверки
    reels_preview.mp4                   все клипы одним файлом
output/_runs/latest.json                отчёт последнего прогона
```

---

## RU: Перерендер из готового moments.json

Поменять кадр, кодек или субтитры можно без нового анализа:

```bash
python3 rerender_videos.py --smart-crop-face --replace
python3 rerender_videos.py --help
```

Обычный запуск сам перенарезает клипы, если изменились настройки `video`,
`subtitles` или фильтры качества: у стадии есть отпечаток входов.

---

## RU: Кадр и субтитры

- **Кадр** (`video.smart_crop_face: true`) весь клип показывает того, кто говорит:
  лица ищет YuNet на GPU, говорящего определяет Light-ASD по губам и звуку.
  Нужна CUDA; без неё клип получает центральный кроп. Отчёт по клипу —
  `reels/framing/reel_XX.json`.
- **Субтитры**: готовый стиль одной строкой — `subtitles.preset: hormozi`
  (или `mrbeast`, `tiktok`, `karaoke`, `neon`…, всего 16). Визуально стиль
  настраивается в GUI: откройте `gui/index.html`, вкладка «Субтитры».
  Все ключи — в [CONFIGURATION.md](CONFIGURATION.md#subtitles--субтитры).

---

## RU: Частые проблемы

- **OOM на транскрипции**: `batch_size` сам делится пополам вплоть до CPU; чтобы
  было быстрее, освободите VRAM (остановите другие GPU-задачи).
- **llama.cpp не отвечает**: смотрите `llama-server.log`; увеличьте
  `llama_cpp.timeout` или таймауты в `llama_cpp.role_overrides`.
- **Кадр показывает не того**: проверьте `reels/framing/reel_XX.json`; попробуйте
  `video.speaker_switch: pan` или `video.active_speaker: false`.
- **Субтитры расходятся с речью**: включите `subtitles.whisper_sync.enabled` и
  посмотрите `reels/subtitle_sync.json`.
- **Диаризация не запускается**: нужен токен Hugging Face (`PYANNOTE_TOKEN` или
  `HF_TOKEN`) и принятые условия моделей pyannote.
- **Прогон завершился с кодом 75**: уже идёт другой прогон (блокировка
  `autonomy.lock_file`).

---

## EN: What it is

Podcast Reels Forge finds strong moments in long-form podcasts and interviews
and cuts them into vertical clips for Reels / Shorts / TikTok. Everything runs
locally: speech recognition, the language model, the editing.

Default pipeline:

0. **fetch** — (when YouTube is given) a video, playlist or channel is downloaded into `input/youtube/`.
1. **transcribe** — `faster-whisper large-v3` with per-word timestamps.
2. **diarize** — (optional) speaker separation with `pyannote`.
3. **proofread** — gemma4 fixes spelling and punctuation.
4. **article** — an episode long-read (`<stem>.article.md`).
5. **analyze** — moment discovery: scout → cleanup → judge on a local Gemma 4 through llama.cpp; every quote is checked against the transcript.
6. **cut** — 9:16 clips: edges at pauses in speech, a frame that follows the speaker, burned-in subtitles.

---

## EN: Quick start

```bash
git clone https://github.com/Bormotoon/Podcast-Reels-Forge.git
cd Podcast-Reels-Forge
python3 -m venv whisper-env
source whisper-env/bin/activate
pip install -r requirements.txt
pip install -e ".[youtube,diarization]"   # optional
cp .env.example .env                       # optional: tokens and keys
```

You need Python 3.10+, FFmpeg (ideally with NVENC and libass), `llama-server`
with a Gemma 4 GGUF model, and an NVIDIA GPU with CUDA. Put machine-specific
settings (the model path, ports) in the gitignored `config.local.yaml`, which is
merged over `config.yaml`.

Drop videos or audio into `input/` (sub-folders are scanned too) and run:

```bash
python3 start_forge.py
python3 start_forge.py --youtube "https://youtu.be/D6WjXRJt1DA" --yt-video
python3 start_forge.py --only analyze,cut
```

Scheduled unattended runs are covered in [AUTONOMOUS.md](AUTONOMOUS.md).

---

## EN: Output layout

```text
output/<stem>/
  <stem>.json, <stem>.srt                 transcript
  <stem>.proofread.json, .proofread.srt   proofread transcript
  <stem>.article.md, .article.json        long-read
  diarization.json                        speakers (when diarization is on)
  gemma4_26b/                             analysis model folder
    moments.json                          final clip list
    reels.md                              clip summary
    rejected_candidates.json              rejected candidates with the reason
    analysis_metrics.json                 analysis run metrics
    reels/reel_XX.mp4                     clips
    reels/reel_XX.md, reel_XX.srt         caption with hashtags, subtitles
    reels/framing/reel_XX.json            who the frame showed and when
    reels/subtitle_sync.json              timing shift after the Whisper re-check
    reels_preview.mp4                     all clips in one file
output/_runs/latest.json                  report of the latest run
```

---

## EN: Re-render existing moments

```bash
python3 rerender_videos.py --smart-crop-face --replace
```

A normal run re-cuts clips by itself when `video`, `subtitles` or the quality
filters change: every stage has an input fingerprint.

---

## EN: Framing and subtitles

- **Framing** (`video.smart_crop_face: true`) shows whoever is talking for the
  whole clip: YuNet finds faces on the GPU, Light-ASD tells the speaker from
  lips and sound. Needs CUDA; without it a clip gets a centre crop. Per-clip
  report: `reels/framing/reel_XX.json`.
- **Subtitles**: a complete look in one line — `subtitles.preset: hormozi` (or
  `mrbeast`, `tiktok`, `karaoke`, `neon`…, 16 in all). Tune it visually in the
  GUI: open `gui/index.html`, Subtitles tab. Every key is in
  [CONFIGURATION.md](CONFIGURATION.md#subtitles--субтитры).

---

## EN: Troubleshooting

- **Transcription OOM**: `batch_size` halves itself down to the CPU; free VRAM to
  keep it fast.
- **llama.cpp stalls**: read `llama-server.log`; raise `llama_cpp.timeout` or the
  per-role timeouts in `llama_cpp.role_overrides`.
- **The frame shows the wrong person**: check `reels/framing/reel_XX.json`; try
  `video.speaker_switch: pan` or `video.active_speaker: false`.
- **Subtitles drift from the speech**: enable `subtitles.whisper_sync.enabled`
  and look at `reels/subtitle_sync.json`.
- **Diarization won't start**: it needs a Hugging Face token (`PYANNOTE_TOKEN` or
  `HF_TOKEN`) and the pyannote model terms accepted.
- **Exit code 75**: another run is active (the `autonomy.lock_file` lock).
