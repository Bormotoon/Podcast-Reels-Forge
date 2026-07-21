# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Episode long-read** — after proofreading, gemma4 edits the transcript into a
  readable article: meaning-based sections with headings, paragraphs, corrected
  errors. It is not a retelling; the speaker's words, phrasing and grammatical
  person are kept verbatim, and only filler, slips and verbatim repetitions are
  removed. Writes `<stem>.article.md` + `.json`.
- **Run any subset of stages** — `start_forge.py --only`/`--skip`/`--list-stages`
  (`transcribe`, `diarize`, `proofread`, `article`, `analyze`, `cut`). An unknown
  stage name is an error, not a silent skip. The GUI offers the same choice as
  checkboxes and assembles the matching command, since the pages run no backend.
- `llama_cpp.roles.article`, the `article` config block, and `json_output` on the
  llama.cpp provider for stages that want prose instead of JSON.
- **Speaker separation in the long-read** — with a `diarization.json` alongside,
  every turn starts with the speaker's name. Speakers are assigned per word (a
  36-second Whisper segment can hold three people), boundaries snap to sentence
  ends, and names are read out of the conversation rather than invented. A label
  whose name is never stated keeps its technical id.
- **MP3 and WAV companions built in one ffmpeg pass**, both decoded from the
  video's own audio. The models read the 16 kHz mono PCM — which is exactly what
  faster-whisper and pyannote resample to internally — instead of the MP3.

### Fixed
- Diarization could not run at all: pyannote reads a file in chunks and raises on
  MP3 because a crop comes back a few samples short of what it requested.
  `diarize.py` now decodes any non-PCM input first.
- Skipping the proofread stage no longer strands the ones after it: the pipeline
  switched to the corrected transcript only inside that stage's block, so
  `--only analyze` would have quietly fed the raw text downstream.
- Chunk units read `str(segment.get("speaker", ""))`, but a transcript without
  diarization stores an explicit `speaker=None` — so `str(None)` prefixed every
  line sent to the model with a literal `(None)`, and the prefixes ate a third of
  the per-chunk character budget.

## [1.3.0] — 2026-07-20

### Added
- **Viral-caption defaults** — the shipped subtitle style is now the format that
  dominates podcast shorts: heavy condensed face, a thick black outline instead
  of a drop shadow, and a `\kf` sweep from white (not yet spoken) to amber
  `#FFD60A` (already spoken), anchored bottom-centre clear of the platform UI
  (`MarginL/R 140`, `MarginV 470`).
- **`fade_in_duration` / `fade_out_duration` now render** as an ASS `\fad` tag.
  They were parsed and then dropped. Fades longer than a cue are scaled down
  proportionally so it still reaches full opacity; `0` disables them.
- **`max_width_ratio` now drives line length.** The 25 chars/line guideline is
  measured at a 0.65 frame share, so the value scales from there (the new 0.74
  default gives ~28).
- **`vertical_offset` now renders** as a per-cue `MarginV` override on top of the
  `.ass` style. `0.0` keeps output byte-identical to before.
- Sentence-split `.srt` output — one short cue per screen instead of a
  three-or-four-sentence block.

### Fixed
- **Subtitle style preview ignored most settings.** The font never loaded in the
  GUI (a hardcoded `../../` 404'd from `gui/`, so the preview silently fell back
  to sans-serif); words rendered glued together because they were adjacent flex
  items whose only separation was `Spacing`, and the inter-word spaces inherited
  a 16px font instead of the caption size; and "primary fill" was invisible
  because it only applies to already-sung words, of which there were none while
  the karaoke simulation was off. A frozen preview now shows the middle of a
  `\kf` sweep so both fills stay visible and tunable.
- Burned reel subtitles now come from the **proofread** transcript on every path:
  `rerender_videos.py` preferred the raw `.json` when auto-detecting, which
  silently undid the proofreading stage on a re-burn.
- `fade_*` of `0` was clamped to `0.01`, so the fade could not be turned off.
- Removed a duplicate `id="dynamic-font-face"` from the subtitles page, guarded a
  platform lookup that could throw, and stopped the 400 ms karaoke tick from
  rebuilding the DOM and re-fetching the font on every frame.

### Removed
- `subtitles.word_x_space` / `word_y_space` controls from the GUI and exported
  config — they never affected rendering, and spacing comes from the `.ass` style
  (`Spacing` in the editor). Still parsed so existing configs keep loading.

## [1.2.0] — 2026-07-19

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
- **The episode overview inherited the moments grammar** and could only answer
  `{"moments": ...}` — `{"summary": ...}` was unrepresentable, so the context
  silently never worked. It now has its own schema and provider.
- **Duration-scaled targets could be starved twice over**: cleanup batches
  overflowed the response token budget and lost candidates (batch size
  25 → 16, and the pool is topped back up from scouted candidates when
  cleanup shrinks it below the target), and per-type quota slots with no
  matching candidates simply vanished (with `clips_per_hour` active, unfilled
  quota now spills over to the best remaining candidates of any type).
- The pipeline progress bar rendered one step behind and finished at 3/4;
  it now advances explicitly and completes.
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

[1.3.0]: https://github.com/Bormotoon/Podcast-Reels-Forge/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/Bormotoon/Podcast-Reels-Forge/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/Bormotoon/Podcast-Reels-Forge/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/Bormotoon/Podcast-Reels-Forge/releases/tag/v1.0.0
