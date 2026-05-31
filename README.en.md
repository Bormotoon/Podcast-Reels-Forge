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

1. **Speech Recognition (Whisper)**: Converts audio/video into text with precise timestamps.
2. **Diarization (Optional)**: Identifies different speakers throughout the audio.
3. **AI Analysis (LLM)**: A staged `scout -> cleanup -> refine -> judge -> metadata` flow on local Gemma models.
4. **Video Editing (FFmpeg)**: Cuts the video, applies vertical cropping (9:16), stabilized face framing, and burns subtitles.

Detailed user guide: [docs/USER_GUIDE.md](docs/USER_GUIDE.md)

---

## Key features

- **Batch Processing**: Drop multiple videos into `input/`, and the forge will process them all sequentially.
- **Role-based llama.cpp pipeline**: Local staged flow through **llama.cpp** with a Gemma 4 lineup: `gemma4`.
- **Smart Face Crop**: Automatically detects faces and centers the frame during vertical cropping.
- **Hardware Acceleration**: Uses **CUDA** for Whisper and **NVENC** for high-speed video rendering.
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

---

## Configuration (config.yaml)

### Key Sections

- **`transcription`**: Choose Whisper model, device (`auto`/`cuda`/`cpu`), and language.
- **`llama_cpp`**:
  - `roles`: Role mapping for `scout / cleanup / refine / judge / metadata`.
  - `role_overrides`: Per-role timeout and chunk-size tweaks.
  - `model_overrides`: Legacy compatibility only, not the primary path.
- **`processing`**: Set counts and durations for clip types (`stories`, `reels`, `highlights`).
- **`video`**:
  - `vertical_crop`: Enable/disable 9:16 aspect ratio.
  - `smart_crop_face`: Enable smart centering on faces.
  - `use_nvenc`: Use NVIDIA hardware acceleration.
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

- **Whisper**: If you hit Out of Memory (OOM) errors, use `small` or `base` models.
- **llama.cpp**: If a model takes too long to respond, increase `llama_cpp.watchdog.first_token_timeout` in config.
- **FFmpeg**: If encoder errors occur, try disabling `use_nvenc` in config to use CPU encoding instead.
- **Pycaps / Playwright**: If subtitle rendering fails with a Chromium-related error, run `playwright install chromium`.

---

## License

MIT License — see [LICENSE](LICENSE).
