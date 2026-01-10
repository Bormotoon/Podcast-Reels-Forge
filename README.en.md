# 🎙️ Podcast Reels Forge

## Automatically create viral clips (Reels/Shorts) from podcasts

## Автоматическое создание вирусных клипов (Reels/Shorts) из подкастов

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**English** | [Русский](README.md)

---

## 📋 Description

**Podcast Reels Forge** is a command-line tool for automatically creating short viral videos (Reels, Shorts, TikTok) from long podcasts and interviews.

### 🎯 What this tool does

1. **Transcribes** audio/video using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (local, GPU/CPU)
2. **Analyzes** the transcript with LLM (Ollama, OpenAI, Anthropic, Gemini) to find the most interesting moments
3. **Cuts** the video into ready clips with automatic cropping to vertical 9:16 format
4. **Exports** to various formats (MP4, WebM, GIF, audio)

### ✨ Key Features

- 🚀 **Single command** — the entire pipeline runs with one command
- 🎛️ **Flexible configuration** — all parameters in a YAML file
- 🤖 **Multi-LLM** — support for Ollama (local), OpenAI, Anthropic, Gemini
- 🌍 **Multilingual** — prompts in Russian and English, auto-detection of language
- 📊 **A/B testing** — built-in prompt quality evaluation
- 🎬 **Vertical format** — automatic crop from 16:9 to 9:16 for social media
- 🔄 **Parallel processing** — multi-threaded video cutting via FFmpeg

---

## 🚀 Quick Start

### Requirements

- Python 3.10+
- FFmpeg (installed system-wide)
- CUDA (optional, for GPU acceleration)
- Ollama (optional, for local LLM)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/podcast-reels-forge.git
cd podcast-reels-forge

# Create virtual environment
python3 -m venv whisper-env
source whisper-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# 1. Put video in input/ folder
cp your_podcast.mp4 input/

# 2. Run the pipeline
python3 start_forge.py

# 3. Results will be in output/ folder
#    - output/reels/         — ready clips
#    - output/moments.json   — moment metadata
#    - output/reels.md       — descriptions for publishing
```

---

## 📁 Project Structure

```text
podcast-reels-forge/
├── start_forge.py              # Main entry point
├── config.yaml                 # Pipeline configuration
├── requirements.txt            # Dependencies
├── input/                      # Input video/audio files
├── output/                     # Processing results
│   ├── reels/                  # Ready clips
│   ├── moments.json            # Found moments
│   └── reels.md                # Clip descriptions
├── prompts/                    # LLM prompts
│   ├── ru/                     # Russian prompts
│   └── en/                     # English prompts
├── podcast_reels_forge/        # Main package
│   ├── pipeline.py             # Pipeline orchestration
│   ├── scripts/                # CLI scripts for stages
│   │   ├── analyze.py          # LLM analysis
│   │   ├── transcribe.py       # Transcription
│   │   ├── video_processor.py  # Video processing
│   │   ├── diarize.py          # Diarization (optional)
│   │   └── evaluate_prompts.py # A/B testing for prompts
│   ├── llm/                    # LLM providers
│   │   └── providers.py        # Ollama, OpenAI, Anthropic, Gemini
│   ├── stages/                 # Stage modules
│   └── utils/                  # Utilities
└── tests/                      # Tests
```

---

## ⚙️ Configuration

All settings are in the `config.yaml` file:

```yaml
# Directory paths
paths:
  input_dir: "input"
  output_dir: "output"

# CLI settings
cli:
  quiet: false      # Errors only
  verbose: false    # Detailed output

# Transcription (faster-whisper)
transcription:
  model: "small"              # tiny, base, small, medium, large-v3
  device: "cuda"              # cuda or cpu
  language: "en"              # ru, en, or auto
  beam_size: 5
  compute_type: "int8_float16"

# LLM provider
llm:
  provider: "ollama"          # ollama | openai | anthropic | gemini
  openai_model: "gpt-4o-mini"
  anthropic_model: "claude-3-5-sonnet-20241022"
  gemini_model: "gemini-1.5-flash"

# Ollama settings (if provider: ollama)
ollama:
  url: "http://127.0.0.1:11434/api/generate"
  model: "gemma2:9b"
  timeout: 900
  temperature: 0.3

# Prompts
prompts:
  language: "en"              # ru | en | auto
  variant: "default"          # default | a | b (A/B testing)

# Processing parameters
processing:
  reels_count: 4              # Number of clips
  reel_min_duration: 30       # Min clip length (sec)
  reel_max_duration: 60       # Max clip length (sec)
  reel_padding: 0             # Padding around moment (sec)

# Export additional formats
exports:
  webm: false
  gif: false
  audio_only: false

# Video settings
video:
  threads: 4
  vertical_crop: true         # Crop to 9:16
  video_bitrate: "5M"
  audio_bitrate: "192k"
  preset: "fast"

# Diarization (optional)
diarization:
  enabled: false
  model: "pyannote/speaker-diarization"
```

---

## 🤖 LLM Providers

### Ollama (local, default)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull gemma2:9b

# Configuration
llm:
  provider: "ollama"
ollama:
  model: "gemma2:9b"
```

### OpenAI

```bash
export OPENAI_API_KEY="sk-..."

# Configuration
llm:
  provider: "openai"
  openai_model: "gpt-4o-mini"
```

### Anthropic (Claude)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."

# Configuration
llm:
  provider: "anthropic"
  anthropic_model: "claude-3-5-sonnet-20241022"
```

### Google Gemini

```bash
export GEMINI_API_KEY="..."

# Configuration
llm:
  provider: "gemini"
  gemini_model: "gemini-1.5-flash"
```

---

## 📊 A/B Testing Prompts

The tool includes a system for comparing quality of different prompts:

```bash
# Run evaluation of all variants
python -m podcast_reels_forge.scripts.evaluate_prompts \
    --transcript output/transcript.json \
    --outdir output \
    --variants default,a,b

# Results in output/prompt_eval.json
```

Prompt files are located in `prompts/ru/` and `prompts/en/`:

- `chunk_default.txt`, `chunk_a.txt`, `chunk_b.txt` — prompts for chunk analysis
- `select_default.txt`, `select_a.txt`, `select_b.txt` — prompts for final selection

---

## 🔧 CLI Options

```bash
python3 start_forge.py --help

# Options:
#   --config CONFIG   Path to config.yaml (default: config.yaml)
#   --quiet           Errors only
#   --verbose         Detailed output
```

### Running Individual Stages

```bash
# Transcription only
python -m podcast_reels_forge.scripts.transcribe \
    --input input/video.mp4 \
    --outdir output \
    --model medium \
    --language en

# Analysis only
python -m podcast_reels_forge.scripts.analyze \
    --transcript output/video.json \
    --outdir output \
    --provider ollama \
    --model gemma2:9b

# Video cutting only
python -m podcast_reels_forge.scripts.video_processor \
    --input input/video.mp4 \
    --moments output/moments.json \
    --outdir output \
    --vertical
```

---

## 🧪 Development

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Type checking
mypy podcast_reels_forge/

# Code formatting
black podcast_reels_forge/ tests/
```

---

## 📦 Output Files

After running the pipeline, the `output/` folder will contain:

| File | Description |
| ------ | ------------- |
| `{video}.json` | Transcript with timecodes |
| `moments.json` | Found viral moments |
| `reels.md` | Markdown with descriptions for publishing |
| `reels/reel_01.mp4` | Ready clips in vertical format |
| `reels_preview.mp4` | Preview of all clips concatenated |
| `diarization.json` | Diarization (if enabled) |

---

## ❓ FAQ

### How to choose a Whisper model?

| Model | VRAM | Accuracy | Speed |
| ------- | ------ | ---------- | ------- |
| `tiny` | ~1GB | Low | Very fast |
| `base` | ~1GB | Below average | Fast |
| `small` | ~2GB | Average | Medium |
| `medium` | ~5GB | Good | Slow |
| `large-v3` | ~10GB | Best | Very slow |

### NVENC not working

If your GPU doesn't support NVENC, the pipeline will automatically fall back to `libx264`.

### LLM returns invalid JSON

The pipeline automatically attempts to "fix" JSON through a follow-up request to the LLM.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgements

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — fast transcription
- [Ollama](https://ollama.com/) — local LLMs
- [FFmpeg](https://ffmpeg.org/) — video processing
- [pyannote-audio](https://github.com/pyannote/pyannote-audio) — speaker diarization
