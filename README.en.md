# 🎙️ Podcast Reels Forge

## Automatically create Reels/Shorts from podcasts (local-first)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**English** | [Русский](README.md)

---

## 📌 Table of contents

- What the project does
- Quick start
- Pipeline overview
- Output layout
- Models and `output/<model>/`
- Re-render video from existing `moments.json` (no LLM)
- Face-aware smart crop
- Running individual stages
- Configuration (`config.yaml`)
- Performance & stability
- Troubleshooting / FAQ
- Documentation

---

## 📋 What it does

**Podcast Reels Forge** is a CLI tool that:
1) transcribes audio/video via `faster-whisper` (CUDA/CPU),
2) finds “viral moments” via an LLM (default: Ollama),
3) cuts clips via FFmpeg,
4) stores outputs **per model** under `output/<model>/`.

Detailed user guide: [docs/USER_GUIDE.md](docs/USER_GUIDE.md)

---

## ✨ Key features

- **5 supported models** (qwen3/deepseek/gemma3/gemma2/gemini3) with separate outputs.
- **Single-pass analysis**: no multi-iteration “boosting”, no repair/selection loops.
- **Resilient**: if one model fails/stalls, the pipeline continues with others.
- **Ollama watchdog** (first-token / stall timeouts) + per-model overrides.
- **Vertical 9:16 cutting** in the pipeline, plus a **strict re-render** script for guaranteed export.
- **Optional face-aware smart crop**: shifts 9:16 crop window toward detected faces.

---

## 🚀 Quick start

### Requirements

- Python 3.10+
- FFmpeg: `ffmpeg` and `ffprobe` available in PATH
- Ollama (for local LLM)

Check:

```bash
ffmpeg -version
ffprobe -version
```

### Install

```bash
python3 -m venv whisper-env
source whisper-env/bin/activate
pip install -r requirements.txt
```

### Prepare input

Put your video/audio into `input/`. The pipeline picks the **newest** matching file.

```bash
cp /path/to/video.mp4 input/video.mp4
```

### Run the full pipeline

```bash
python3 start_forge.py
```

Useful flags:
- `--verbose` for detailed logs
- `--quiet` for errors-only
- `--no-skip-existing` to rerun stages even if outputs exist

---

## 🔩 Pipeline overview

Entry: [start_forge.py](start_forge.py) → [podcast_reels_forge/pipeline.py](podcast_reels_forge/pipeline.py)

1) **Transcribe** → `output/<input_name>.json` (segments + timecodes)
2) **Analyze** (per model) → `output/<model>/moments.json` and `output/<model>/reels.md`
3) **Cut** (per model) → `output/<model>/reels/reel_XX.mp4` + `reels_preview.mp4`

Important behavior:
- The analyzer expects the LLM to return **valid JSON** (no second “repair” call).
- There is **no** final LLM selection pass by default (top moments are chosen locally by `score`).

---

## 📁 Output layout

### Transcript

- `output/<input_name>.json` — transcription output.

### Per-model outputs

For each model folder (e.g. `output/gemma3/`):

- `moments.json` — extracted moments (start/end/title/quote/why/score/...)
- `reels.md` — text snippets/metadata
- `reels/` — cut clips `reel_01.mp4`, `reel_02.mp4`, ...
- `reels_preview.mp4` — concatenated preview

Model folder names are stable and short:
- `qwen3`, `deepseek`, `gemma3`, `gemma2`, `gemini3`

---

## 🎞️ Re-render video from existing moments.json (no LLM)

This is the best way to fix format/quality without re-running analysis.

Root launcher: [rerender_videos.py](rerender_videos.py)

Examples:

```bash
# One model
python3 rerender_videos.py --model gemma3

# All models under output/
python3 rerender_videos.py
```

Default behavior:
- reads `output/<model>/moments.json`
- writes to `output/<model>/reels_rerendered/`
- **never overwrites**: if a filename exists, it creates `reel_01_2.mp4`, `reel_01_3.mp4`, ...

Default export settings:
- 1080x1920
- 30fps
- H.264 (`libx264`)
- AAC
- `-b:v 5000k`

To overwrite into `reels/`:

```bash
python3 rerender_videos.py --model gemma3 --replace
```

---

## 🙂 Face-aware smart crop (optional)

Enable for re-render:

```bash
python3 rerender_videos.py --model gemma3 --smart-crop-face
```

Tuning:
- `--face-samples 7` — how many frames to sample per clip
- `--face-min-size 60` — minimum detected face size

Notes:
- Uses OpenCV Haar cascade (fast, offline).
- Falls back to center-crop when no face is detected.

---

## 🧩 Running individual stages

### Transcription only

```bash
python -m podcast_reels_forge.scripts.transcribe \
	--input input/video.mp4 \
	--outdir output \
	--model medium \
	--language en
```

### Analysis only (single model)

```bash
python -m podcast_reels_forge.scripts.analyze \
	--transcript output/video.json \
	--outdir output/gemma2 \
	--provider ollama \
	--model gemma2:9b \
	--timeout 600 \
	--chunk-seconds 600
```

### Cutting only (pipeline cutter)

```bash
python -m podcast_reels_forge.scripts.video_processor \
	--input input/video.mp4 \
	--moments output/gemma3/moments.json \
	--outdir output/gemma3 \
	--vertical \
	--padding 5
```

### Prompt A/B evaluation

```bash
python -m podcast_reels_forge.scripts.evaluate_prompts \
	--transcript output/video.json \
	--outdir output \
	--variants default,a,b \
	--provider ollama \
	--model gemma2:9b
```

---

## ⚙️ Configuration (config.yaml)

File: [config.yaml](config.yaml)

Most important sections:

### `ollama.models` + per-model overrides

- The pipeline analyzes all models listed in `ollama.models`.
- For slow models, use `ollama.model_overrides` to adjust timeouts and watchdog thresholds.

### Watchdog

If a model is slow to produce the first token, increase:
- `ollama.watchdog.first_token_timeout`

If a model stalls mid-stream, increase:
- `ollama.watchdog.stall_timeout`

---

## ⚡ Performance & stability

- If everything is slow: increase `ollama.chunk_seconds` (fewer requests), but watch quality.
- If qwen3/deepseek need warm-up: use `ollama.model_overrides` with bigger `first_token_timeout`.
- If Whisper hits CUDA OOM: reduce `transcription.model` or set `transcription.device: cpu`.

---

## ❓ Troubleshooting / FAQ

### Output video is horizontal

- For the pipeline, ensure `video.vertical_crop: true` in [config.yaml](config.yaml).
- For guaranteed 9:16 + consistent export, use `python3 rerender_videos.py`.

### LLM returned invalid JSON → empty moments

The pipeline **does not** do a second “repair” LLM call.
Options:
- try another model,
- reduce chunk size (`ollama.chunk_seconds`),
- adjust prompts (see [docs/PROMPTS.md](docs/PROMPTS.md)).

### FFmpeg not found

Install FFmpeg (Ubuntu: `sudo apt install ffmpeg`).

---

## 📚 Documentation

- [docs/USER_GUIDE.md](docs/USER_GUIDE.md) — user guide
- [docs/CONFIGURATION.md](docs/CONFIGURATION.md) — configuration
- [docs/PROMPTS.md](docs/PROMPTS.md) — prompts
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) — development

---

## 📄 License

MIT License — see [LICENSE](LICENSE).
