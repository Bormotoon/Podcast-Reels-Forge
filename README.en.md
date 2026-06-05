# 🎙️ Podcast Reels Forge

## Automatically create Reels/Shorts from podcasts (local-first)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**English** | [Русский](README.md)

---

## 📌 Table of Contents

- [What it does](#what-it-does)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Run Modes by Task](#run-modes-by-task)
- [Pipeline Overview](#pipeline-overview)
- [Output Layout](#output-layout)
- [Command Line Arguments](#command-line-arguments)
- [Configuration (config.yaml)](#configuration-configyaml)
- [Face-Aware Smart Crop](#face-aware-smart-crop-optional)
- [Rerendering Videos](#re-render-video-from-existing-momentsjson)
- [Performance and Stability](#performance-and-stability)
- [License](#license)

---

## What it does

**Podcast Reels Forge** is a powerful CLI tool designed to automatically extract viral short-form content (Reels, Shorts, TikTok) from long-form podcasts or interviews. It handles everything from speech recognition to final video editing.

Main workflow steps:

1. **Speech Recognition (faster-whisper)**: Converts audio/video into text with precise timestamps. Model `large-v3`, with two modes — fast batched and accurate context-aware (see [Run Modes by Task](#run-modes-by-task)).
2. **Diarization (Optional)**: Identifies different speakers throughout the audio.
3. **AI Analysis (LLM)**: A staged `scout -> cleanup -> refine -> judge -> metadata` flow on local Gemma models.
4. **Video Editing (FFmpeg + NVENC)**: Cuts the video, applies vertical cropping (9:16), stabilized face framing, and burns subtitles. GPU encoding via NVENC (~5× faster than software).

Detailed user guide: [docs/USER_GUIDE.md](docs/USER_GUIDE.md)

---

## Key features

- **Batch Processing**: Drop multiple videos into `input/`, and the forge will process them all sequentially.
- **Two transcription modes**: `fast` (batched, ~5 min per hour of audio) and `quality` (sequential with context, more accurate on quiet/noisy recordings).
- **Hallucination guard**: Suppresses Whisper repetition loops (endless "Thank you." on silence/music) via a temperature ladder, repetition penalty, and `condition_on_previous_text`.
- **Role-based llama.cpp pipeline**: Local staged flow through **llama.cpp** with a Gemma 4 lineup: `gemma4`.
- **Smart Face Crop**: Automatically detects faces and centers the frame during vertical cropping.
- **Hardware Acceleration**: **CUDA** (ctranslate2) for Whisper and **NVENC** for video rendering. The NVENC-capable ffmpeg is auto-detected.
- **llama.cpp Watchdog**: Monitors model responsiveness and automatically retries stalled generations.
- **Flexible Clip Types**: Configure specific counts and durations for Stories, Reels, and Highlights separately.

---

## Quick start

### Requirements

- **Python 3.10+**
- **FFmpeg** (must be in PATH)
- **llama.cpp (`llama-server`)** (for local LLM support)
- **NVIDIA GPU** (highly recommended for performance)

### Installation

1. Clone the repository.
2. Create a virtual environment and install dependencies:

```bash
python3 -m venv whisper-env
source whisper-env/bin/activate
pip install -r requirements.txt
playwright install chromium
```

### Prepare Input

Place your video files (mp4, mkv, mov) in the `input/` directory.
*Tip: If a same-name `mp3` already exists, Forge will use it. Otherwise it automatically extracts audio from the video into `video.mp3` at 320 kbps and continues the pipeline as usual.*

### Run

```bash
python3 start_forge.py
```

---

## Run modes by task

Forge can run end-to-end (full pipeline) or transcription-only. Transcription has two modes — `fast` (default) and `quality`.

| Mode | When to use | Speed\* |
|---|---|---|
| `fast` (batched) | Clean recording, need a quick draft | ~5 min per hour of audio |
| `quality` (sequential, context-aware) | Quiet/far-field/noisy recording (dictaphone, phone, hall): fixes garbled words | ~1 h per hour of audio |

\* Reference for an RTX 5060 Ti 16GB with `large-v3`. Quality mode is slower because it processes segments sequentially with language context instead of independent batches.

### Full pipeline (transcribe → analyze → cut with NVENC)

```bash
python3 start_forge.py                     # transcription mode comes from config.yaml
python3 start_forge.py --verbose           # verbose logs
python3 start_forge.py --no-skip-existing  # rerun all stages, ignore cache
```

For the full pipeline, set the transcription mode in `config.yaml` → `transcription.mode` (`fast`/`quality`).

### Transcription-only for audio in `input/`

```bash
# Fast mode (default)
python3 transcribe_input_audio.py --verbose

# Quality mode + topic hint — greatly improves quiet/noisy recordings
python3 transcribe_input_audio.py --verbose --mode quality \
  --initial-prompt "School parent meeting. Curriculum, classes, teachers."

# Re-transcribe from scratch, ignoring cache
python3 transcribe_input_audio.py --verbose --no-skip-existing
```

> 💡 **Topic hint** (`--initial-prompt`) biases the model's vocabulary and helps in both modes. Provide context specific to the recording.

> 💡 **Audio denoising** in practice **hurts** recognition — Whisper is trained on "dirty" audio. Get gains from quality mode and the topic hint, not from preprocessing.

### Re-render videos without AI analysis

```bash
python3 rerender_videos.py --smart-crop-face --replace
```

### Inspect the result

```bash
python3 - <<'PY'
import json; d=json.load(open('output/<stem>/<stem>.json'))
s=d['segments']
print('mode:', d.get('mode'), '| segments:', len(s))
PY
```

---

## Pipeline overview

The orchestrator [start_forge.py](start_forge.py) runs [podcast_reels_forge/pipeline.py](podcast_reels_forge/pipeline.py), which executes the following stages for each file:

1. **Transcription**: Uses `faster-whisper`. Output: `output/<file_stem>/audio.json` + `audio.srt`.
2. **Diarization**: (If enabled) Creates `diarization.json` with speaker turns.
3. **Analyze (Staged)**: One final pass over Gemma roles. By default, artifacts are written into `output/<file_stem>/gemma4/`.
4. **Video Processing**: Cuts clips from the final `moments.json`. Forge burns subtitles into each reel with `pycaps`, adds a ready-to-post `reel_XX.md`, keeps a local `reel_XX.srt`, and builds `reels_preview.mp4`.


---

## Output layout

Inside the `output/` directory:

```text
output/
  my_podcast/
    video.json            # Transcript
    diarization.json      # (Optional) Speaker info
    gemma4/               # Final judge-model folder
      analysis_manifest.json
      scout_candidates.json
      cleaned_candidates.json
      refined_candidates.json
      judge_report.json
      moments.json        # Final moment list
      reels.md            # Clip summary
      reels/              # Cut video clips .mp4
        reel_01.srt       # Local subtitle timeline used by pycaps
        reel_01.md        # Description + 5 hashtags for reel_01.mp4
      reels_preview.mp4   # Concatenated preview of all clips
```

---

## Command line arguments

Main flags for `start_forge.py`:

- `--config <path>`: Path to config (default: `config.yaml`).
- `--verbose`: Verbose output for all commands and logs.
- `--quiet`: Errors-only mode.
- `--no-skip-existing`: Rerun all stages even if files already exist (ignore cache).
- `--autotune`: Automatically tune parameters for current hardware (smaller chunks, longer timeouts).
- `--no-progress`: Disable progress bar (useful for CI/logging).

Flags for the standalone transcriber `transcribe_input_audio.py`:

- `--mode <fast|quality>`: Override `transcription.mode` from config.
- `--initial-prompt "<text>"`: Topic hint to bias the model's vocabulary.
- `--verbose` / `--quiet`: Log level.
- `--no-skip-existing`: Re-transcribe even if a JSON already exists.

---

## Configuration (config.yaml)

### Key Sections

- **`transcription`**: Whisper model (`large-v3`), device (`auto`/`cuda`/`cpu`), language.
  - `mode`: `fast` (batched) or `quality` (sequential, more accurate).
  - `batch_size`: batch size in fast mode (default 16; on OOM it auto-halves down to CPU).
  - `quality_beam_size`: beam width in quality mode (default 10).
  - `initial_prompt`: default topic hint (overridable with `--initial-prompt`).
- **`llama_cpp`**:
  - `roles`: Role mapping for `scout / cleanup / refine / judge / metadata`.
  - `role_overrides`: Per-role timeout and chunk-size tweaks.
  - `model_overrides`: Legacy compatibility only, not the primary path.
- **`processing`**: Set counts and durations for clip types (`stories`, `reels`, `highlights`).
- **`video`**:
  - `vertical_crop`: Enable/disable 9:16 aspect ratio.
  - `smart_crop_face`: Enable smart centering on faces.
  - `use_nvenc`: Prefer NVIDIA hardware encoding (NVENC). Falls back to libx264 automatically if no NVENC ffmpeg build is found.
  - `nvenc_cq`: NVENC VBR quality target (lower = better; default 21).
  - `nvenc_preset`: NVENC preset `p1`(faster)…`p7`(higher quality), default `p5`.
  - `video_bitrate`: Bitrate ceiling (for NVENC, caps the VBR peak).
- **`subtitles`**:
  - `enabled`: Toggle automatic burned subtitle rendering with `pycaps`.
  - `font`: Path to the subtitle font file. Default: `assets/fonts/bignoodletoooblique.ttf`.
  - `css`: Path to the subtitle CSS template. Default: `assets/subtitles/forge_subtitles.css`.
  - `wrap_words`: Toggle word wrapping for captions. When disabled, the caption stays on one line.
  - `font_size_px`, `max_lines`, `vertical_align`, `vertical_offset`: Fine-tune subtitle styling/layout.
  - Use `assets/subtitles/style-editor.html` for WYSIWYG style tuning; after selecting the project root, the Apply buttons write the current settings straight into `config.yaml` and `assets/subtitles/forge_subtitles.css`.
- **`diarization`**: Enable and configure speaker detection (requires HuggingFace token).

---

## Face-aware smart crop (optional)

When `smart_crop_face: true` is enabled in config:

1. Several frames are sampled from each clip.
2. OpenCV Haar cascades are used to detect faces across multiple sample points.
3. If faces are found, the 9:16 window is shifted with median smoothing so the speaker does not jitter around.
4. If no faces are found, it falls back to a stable center crop with an explicit fallback log.

---

## Re-render video from existing moments.json

If you want to change video parameters (bitrate, crop, padding) without re-running the long AI analysis, use [rerender_videos.py](rerender_videos.py):

```bash
# Re-render everything with smart crop enabled
python3 rerender_videos.py --smart-crop-face --replace
```

---

## Performance and stability

- **Whisper (memory)**: On OOM, `batch_size` auto-halves (16 → 8 → … → CPU), so it won't crash — just slower. To speed up, free VRAM or lower `batch_size`.
- **Whisper (quality)**: Garbled words on quiet/far-field recordings are fixed by `quality` mode + `--initial-prompt`, **not** by audio cleanup (denoising hurts recognition).
- **Blackwell GPU (RTX 50xx)**: Requires `torch>=2.7` built for CUDA 12.x. The PyTorch `sm_120` warning is harmless — Whisper inference runs via ctranslate2, not PyTorch kernels.
- **ffmpeg / NVENC**: Forge auto-detects an NVENC-capable ffmpeg (`/usr/local/bin`, `/usr/bin`); you can force a path via the `FORGE_FFMPEG` env var. If NVENC is unavailable, encoding falls back to CPU (libx264).
- **llama.cpp**: If a model takes too long to respond, increase `llama_cpp.watchdog.first_token_timeout` in config.
- **Pycaps / Playwright**: If subtitle rendering fails with a Chromium-related error, run `playwright install chromium`.

---

## License

MIT License — see [LICENSE](LICENSE).
