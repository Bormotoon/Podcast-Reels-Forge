# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Duration-scaled clip counts** — `processing.clips_per_hour` (default 10)
  targets `round(total hours × N)` clips per episode, computed from the total
  runtime; the `clips` counts become the type mix rather than absolute
  numbers. The cleanup and judge stages automatically batch their LLM calls
  when the target exceeds what one ctx-8192 prompt fits, so large targets are
  actually reachable. `0` restores the fixed counts.
- **Transcript proofreading stage** — gemma4 fixes spelling and punctuation
  before analysis, guarded by a letter-content similarity check that rejects any
  correction that adds, drops or paraphrases text. Writes
  `<stem>.proofread.json`; the raw transcript is left untouched.
- **Quote verification** — a candidate's quote is matched against the words
  actually spoken in its span. Invented quotes are penalized, and a confident
  match widens the clip to contain it.
- **Boundary snapping** — clip bounds are anchored to real sentence (else word)
  boundaries, so clips stop opening and closing mid-sentence.
- **Audio signals** — per-candidate loudness and pause density measured with
  ffmpeg, plus speech rate from the word timings, feeding the ranking.
- **Episode context** — one LLM call summarizing the episode, injected into the
  scout prompt so locally striking moments can be told from episode-level ones.
  Cached in `episode_context.json`.
- **Judge context** — the judge now receives each clip's real opening and closing
  words, which it needs for the hook/ending criteria it was already asked to apply.
- **Topic diversity** in final selection, so a set is not four clips about the
  same thing.
- **Golden-set evaluation** — `evaluate_prompts` scores variants by recall and
  precision against hand-labelled reference moments (`golden/<episode>.json`),
  with `must` moments tracked separately. Workflow in `docs/DEVELOPMENT.md`.
- `processing.analysis` config block for all of the above (see
  `docs/CONFIGURATION.md`); every key is optional.
- Speaker-count pinning for diarization (`diarization.num_speakers`).

### Fixed
- **The mid-thought penalty was never applied.** It was computed and reported,
  but `combined_priority_score` never subtracted it, so nothing guarded against
  clips that open or close mid-sentence.
- **A clip-type quota of `0` meant "unlimited" instead of "none",** so setting
  `stories: count: 0` admitted an unbounded number of story clips.
- **A single failed chunk aborted the whole analysis,** leaving the episode with
  no clips at all. Chunks now fail independently; only a total outage raises.
- **The candidate cap ran before de-duplication,** so a moment found in two
  overlapping chunks consumed two of the 25 slots and pushed out unique finds.
- **Timecodes and stage output were unvalidated.** Candidates are clamped to the
  chunk they were found in and to the episode; cleanup/judge records that overlap
  none of their input are dropped as invented.
- Truncated JSON from a model that hit its token budget no longer discards the
  whole response — everything emitted before the cut is recovered.
- Unparseable JSON is retried instead of silently costing a chunk.
- **Burned subtitles interpolated word timing from character counts** while the
  transcript carried real per-word timestamps all along; the karaoke drifted
  against the speech.
- Diarization works again on pyannote.audio 4.x, and now uses the GPU.
- Analysis artifacts are written group-readable rather than `0600`.
- `llama_cpp.n_predict` is configurable (was hardcoded to 2048), and the scout
  prompt no longer sends each chunk's transcript twice, which had been forcing
  llama.cpp to truncate the prompt at `ctx_size=8192`.

### Changed
- **GUI redesigned** on a token-based design system: dark "forge" theme with an
  ember brand accent and harmonized per-page stage accents, Golos Text +
  JetBrains Mono (Cyrillic-native) in place of Roboto, layered shadows and a
  concentric radius scale, visible keyboard-focus rings on every control,
  `prefers-reduced-motion` support, Firefox slider styling, tabular numerals
  for all live numbers. On narrow screens the navigation rail becomes a top
  bar instead of disappearing. Fixed a long-standing bug where the tooltip
  helper overrode the run button's `position: fixed`, so the dashboard FAB was
  never actually floating. The `--md-sys-color-*` token names, element IDs,
  i18n keys and app.js are untouched — config generation behaves exactly as
  before.
- **`score` now always carries the model's 1-10 rating.** Ranking previously
  overwrote it with its own combined value, which broke
  `processing.quality_filters.min_score` (documented as the 1-10 scale) and fed
  the combined total back into itself on the second ranking pass. The ranking
  value moved to a separate `priority` field, exposed alongside `score` in
  `moments.json` and `reels.md`. **Existing `min_score` thresholds may need
  retuning**, since they were being compared against the wrong scale.
- Scout/cleanup/judge prompts rewritten with explicit virality criteria, an
  anchored score rubric, clip-type guidance and a verbatim-quote requirement
  (ru and en, `_default` only — the `a`/`b` variants are unchanged).
- llama.cpp receives the actual moments JSON schema as a sampling grammar
  instead of `{"type": "object"}`, with an automatic downgrade for builds that
  reject it.
- Scoring factor weights are configurable via `processing.analysis.scoring.weights`.
- Reference model switched to gemma4:26b IQ4_XS with full GPU offload.

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
