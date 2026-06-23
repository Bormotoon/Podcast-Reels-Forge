# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] — 2026-06-23

### Added
- **Browser GUI** — a full, server-less Material Design 3 interface (`gui/`) for
  building `config.yaml` and tuning subtitles, split into separate per-stage pages
  (Dashboard, Transcribe, Analyze, Cut, Subtitles, Settings, Logs) each with its
  own accent colour.
- **Bilingual UI (RU/EN)** with a per-page language toggle; every user-facing
  string is internationalised, including the embedded style editor.
- **Embedded visual ASS style editor** on the Subtitles page, with a live phone
  preview, viral-creator presets, and one-click "Save ASS File".
- GUI coverage of advanced `llama.cpp` settings: service/VRAM tuning, per-role
  overrides (`role_overrides`), `extra_args`, scout parallelism, and clip-type mix.
- Packaging metadata (`pyproject.toml`), `.env.example`, and community health
  files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue/PR
  templates, and Dependabot.

### Changed
- **Subtitles rendering migrated from CSS/pycaps to native ASS** burned in via
  ffmpeg's `ass` filter; the style lives in `assets/subtitles/forge_subtitles.ass`.
- **Face detection switched from OpenCV Haar cascades to MediaPipe** for smart crop.
- **LLM analysis consolidated from 5 stages to 3** and made async via `aiohttp`.
- Aggressive VAD to further suppress Whisper hallucinations.
- Default subtitle `font_size_px` raised from 36 to 96 (readable viral-caption size).
- READMEs (RU + EN) and `docs/` updated to match the current pipeline; added a
  GUI section in both languages.
- CI reworked into a fast lint job plus a `3.10`/`3.12` test matrix with pip caching.

### Fixed
- `scripts/rerender_videos.py` crashed unconditionally after the CSS→ASS migration
  (stale `css_path` field); removed the dead code path.
- Type-safety and lint fixes across `video_processor.py`, `providers.py`,
  `analyze_stage.py`, and `burned_subtitles.py`.
- `LICENSE` is now detected as MIT by GitHub (removed the leading Markdown heading).

### Removed
- Chromium/pycaps subtitle path and the dead `proofread` config section.
- Internal integration-plan document not relevant to public users.

## [1.0.0] — 2026-06-05

Initial public release: Blackwell-GPU support, faster and more accurate
transcription (faster-whisper `large-v3` with fast/quality modes), local
llama.cpp analysis, and NVENC-accelerated 9:16 video rendering.

[1.1.0]: https://github.com/Bormotoon/Podcast-Reels-Forge/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/Bormotoon/Podcast-Reels-Forge/releases/tag/v1.0.0
