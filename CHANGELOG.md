# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Two speakers are stacked, not cropped between.** When two people are
  steadily in frame, `video.two_speaker_layout: split` (default) puts one
  above the other; before, the crop landed on the empty middle between them.
  Small faces in wide shots are also searched for in each half of the frame,
  faces under `face_min_size` (posters, screens) are ignored — the setting was
  read and never used — and the MediaPipe model downloads itself on first use
  instead of the smart crop silently switching off on a fresh install.
- **YouTube metadata is used.** The `.info.json` yt-dlp writes (title,
  description, chapters, tags) now feeds the episode overview, gives the scout
  the author's chapters with timestamps, gives proofreading a glossary of
  names and terms as the author spells them, and gives the article its real
  title. The analysis fingerprint covers it.
- **Rendered clips are checked** with ffprobe (video and audio streams,
  expected duration; optionally mostly-black frames with
  `video.qa_blackdetect`). A broken clip goes to `reels/rejected/` and the cut
  is reported as failed.
- `evaluate_prompts` reports each variant's analysis metrics (quote match,
  survival, duplicates, diversity, quota fill, timing).
- **YouTube as a source** — a new first stage, `fetch`, pulls a video by link, a
  whole playlist or an entire channel into `input/youtube/`, after which the file
  is indistinguishable from one placed by hand. Because it is an ordinary stage,
  `--only` / `--skip` already compose with it: `--youtube "@channel" --skip cut`
  runs everything but the cutting, `--only fetch` just fills the folder.
  `--yt-list` shows what would be taken without downloading. Also available on
  its own as `python -m podcast_reels_forge.scripts.fetch_youtube`.
  - Files are named `YYYY-MM-DD - Title [id]`: the date sorts episode folders
    chronologically, and the id recognises an already-downloaded video even after
    it was retitled on YouTube. A repeat run checks the file on disk first, then
    the `.archive.txt` archive, so a nightly channel run takes only new episodes.
  - Only the audio track is fetched by default (`youtube.download: audio`): an
    hour of podcast is ~64 MB instead of ~1 GB, which is what makes a
    whole-channel run practical. There is then nothing to cut, and the cut stage
    says so rather than failing. `--yt-video` (or `download: video` / `auto`)
    fetches the picture, capped at 1080p — the output is a 1080-wide vertical
    clip either way.
  - `--youtube` narrows the run to the videos it names. Without that, one
    YouTube link would drag along every local episode in `input/` and transcode
    audio for each; `--yt-all-inputs` lifts it.
  - **Exclusions** (`youtube.exclude`, `--yt-exclude`) keep part of a channel out
    entirely — never downloaded, never processed. Naming a playlist rather than
    ids keeps "everything except this show" true on its own as episodes are added
    to it. An exclusion outranks a direct link and is applied before the cap, so
    `--yt-limit 5` returns five usable videos rather than five minus the excluded
    ones. `--yt-exclude` adds to the config list instead of replacing it: a
    standing rule must not be lifted silently by one command.
  - `YOUTUBE_API_KEY` is optional (yt-dlp can do the listing), and travels as an
    `X-goog-api-key` header rather than in the query string, so `--verbose` does
    not log it. Listing a whole channel costs ~41 units of the free daily quota;
    the 100-unit `search.list` is only a last resort for an unresolvable handle.
  - A failed download earns **one retry through other YouTube clients**
    (`web_safari`, `tv`, `android`). For some older videos the default clients
    see no formats at all and the video reads as "This video is not available",
    though the API reports it public and unrestricted. Only after a failure:
    forcing that client set on everything would trade what works for the unknown
    (the `tv` client returns some formats DRM-protected). A rights-holder block
    or region lock is beyond any client, and is reported without stopping the
    queue. `youtube.ydl_options` passes raw yt-dlp options through for the next
    time YouTube changes something, so it need not become a code change.
  - `yt-dlp` is an optional dependency (`pip install -e ".[youtube]"`); a missing
    one yields the install command instead of a traceback.
- **Host RAM can be freed for the duration of a run** (`host_memory`, off by
  default). The pipeline holds several gigabytes on the host, and on a machine
  that also runs VMs that is enough to reach the OOM killer — which does not
  necessarily kill the pipeline: it picks whatever has the highest
  `oom_score_adj`, and a snap-packaged VS Code volunteers itself at 300.
  - Memory is reclaimed live through virtio-balloon, sized from what each guest
    reports as actually used rather than from a number typed months ago.
    `virsh suspend` was not an option: it stops the vCPUs while qemu keeps every
    page. Neither was `managedsave`, which libvirt refuses for a domain with an
    assigned PCI device.
  - **Ballooning frees nothing on a VM with PCI passthrough**, measured rather
    than assumed: its whole guest memory is locked into RAM for IOMMU DMA, so the
    balloon moves, the guest reports free pages, and qemu's RSS does not drop by
    a byte. Such domains are detected via `VmLck` and skipped — squeezing them
    only starves the guest. `host_memory.stop_domains` shuts them down instead
    and starts them again afterwards.
  - Returning the memory is the guaranteed part: original sizes are fsynced to a
    state file *before* the first change, so a normal exit, an exception, Ctrl+C
    and `kill` all restore immediately, and a SIGKILL or OOM is recovered by the
    next run or by `python -m podcast_reels_forge.scripts.host_memory --restore`.
    A restore that fails keeps its record rather than dropping it, so a VM cannot
    be left shrunken with nothing tracking it.
- **The input folder is scanned recursively**, and an episode may now be audio
  with no video at all — that is what an audio-only fetch leaves behind. Every
  stage but cutting works from the audio anyway, and cutting says so plainly
  instead of failing.
- **`.env` in the project root is finally read.** The repo has always shipped
  `.env.example` and git-ignored `.env`, but nothing loaded it, so a
  `PYANNOTE_TOKEN` written there never reached the process. A real environment
  variable still wins over the file.
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

- **Rare-name spell check** (`proofread.terms`, off by default) — the transcript
  said "Курократ" where the person is "Курокрад", and nothing in the episode
  could settle it. Instead of asking what is correct, it checks which spelling an
  outside source knows at all: Wiktionary has 9 hits for one and none for the
  other. A fix lands only when the original is unknown and a variant is known
  with confidence; a failed lookup is kept distinct from zero hits so a flaky
  network cannot rewrite text. Only the suspect word and one neighbour ever
  leave the machine, and every edit is recorded with its evidence.

### Added
- **Unattended runs** (see `docs/AUTONOMY_REVIEW.md`):
  - every stage of every episode runs in a failure guard — a broken file, a
    CUDA OOM, a failed diarization or cut costs that episode, never the rest
    of the queue; audio companions are built per episode instead of for the
    whole queue up front;
  - a run report (`<output_dir>/_runs/<time>.json` and `latest.json`) with the
    status, timing and error of each stage of each episode, and exit codes a
    scheduler can act on: 0 ok, 3 partial, 1 fatal, 75 another run is active;
  - a single-run lock (`autonomy.lock_file`), so a timer and a manual start
    cannot kill each other's llama-server;
  - a preflight check before any work: ffmpeg, faster-whisper, llama-server
    and its model (unless a server already answers), pyannote and
    `PYANNOTE_TOKEN`, yt-dlp, free disk space (`autonomy.min_free_disk_gb`);
    `--skip-preflight` bypasses it;
  - a daily-rotated `logs/forge.log` at INFO whatever the console verbosity,
    and a notification hook (`autonomy.notify`: a shell command with
    `FORGE_*` variables and/or a JSON webhook), on failure or always.

### Changed
- Dependencies are bounded below the next major version, so an unattended
  run never picks up a breaking release on its own; yt-dlp instead updates
  itself every `youtube.self_update_days` days (`youtube.self_update`).
- New guide for scheduled runs: `docs/AUTONOMOUS.md` (systemd timer and cron,
  exit codes, the run report, logs, notifications). The GUI config template
  carries every new key.
- **The queue runs stage by stage** (`autonomy.scheduling: stage`, the
  default): every transcription with one Whisper load, then one llama-server
  session for every LLM stage, then every cut. Before, both models were
  reloaded for each episode (~14 GB of weights per llama-server start).
  `scheduling: episode` keeps the old order. llama-server is no longer started
  when no LLM stage is selected.
- **Proofreading returns only the segments it changed** instead of rewriting
  the whole transcript — the most expensive LLM output of a run. With
  `proofread.scope: clips` it runs after the analysis and only over the
  selected clips' spans (what subtitles and captions show); the article
  forces the full scope.
- **Stage fingerprints** (`.forge_state.json` per episode): the analysis is
  redone when the transcript, processing config, prompts or roles change, and
  the cut when the moments, subtitle transcript, video/subtitle/export
  settings or quality filters change — the old reels are discarded first.
  Outputs made before fingerprints existed are adopted, not redone.
- **LLM answers are cached** per analysis folder (`analysis.llm_cache`): a
  re-run after a crash replays what it already has; unused entries are
  pruned. `analysis_metrics.json` reports the hits.
- Clips the quality filters reject are listed in `reels/rejected.json`
  instead of being encoded (`quality_filters.render_rejected: true` restores
  that); a failed encode now fails the cut, so the run report shows it.
- Subtitle burning uses NVENC when the libass-capable ffmpeg has it.
- The 320k MP3 listening copy is optional (`audio.listening_copy`), the 16 kHz
  WAV can be deleted after the analysis (`audio.delete_wav_after_analysis`),
  and scout/cleanup/judge get their own `n_predict` in `role_overrides`.
- **Moment analysis: "LLM discovers → Python proves → deterministic selector
  chooses → LLM writes metadata"** (from the moment-selection audit in
  `ANALYSIS_REPORT.md`).
  - **The quote is now evidence, not decoration.** It is looked up verbatim
    first (contiguous, normalized), then with a bounded word-level fuzzy match
    — no longer a character-level similarity that a window of frequent short
    words could satisfy. A candidate below `quote_verification.min_ratio` is
    rejected right after the scout, before any cleanup/judge tokens are spent;
    below the new `min_final_ratio` (0.75) it cannot enter the final cut.
    Everything rejected goes to `rejected_candidates.json` with the reason.
    `moments.json` gains `quote_match_method`, `quote_start`/`quote_end`, and
    a final check keeps the quote inside the clip after snapping and clamping.
  - **Each stage has its own schema.** The scout returns only evidence —
    interval, verbatim quote, a one-line `evidence`, `reason_codes`, score — no
    titles, captions or hashtags. A candidate without a quote is dropped.
    Python assigns stable `candidate_id`s.
  - **Cleanup and judge answer with decisions by `candidate_id`** (keep / drop
    / merge; score, title, hook, why) instead of echoing full records. Quotes
    and timecodes are restored from the source record, so the model cannot move
    a clip or rewrite its quote, and output tokens drop sharply. Legacy
    full-record answers are still accepted and traced back to their source.
  - **Judge batches are stratified**: candidates are dealt round-robin by
    priority, so every call sees the same quality spread and scores stay
    comparable; cleanup and judge batches run in parallel
    (`llama_cpp.stage_parallelism`).
  - **Final selection is MMR** (`diversity.mmr_lambda`) instead of a
    sort-then-defer pass, deduplication also compares quotes, and clips may
    overlap by up to `selection.max_overlap_ratio` (20%) of the shorter one
    instead of not at all. The clip type is assigned by Python from the final
    duration when the model's type does not fit.
  - Metadata finalization never fabricates a quote (it used to copy the hook);
    fields generated by code are listed in `derived_fields`.
  - The scout gets clip lengths and a per-chunk target range
    (`{target_candidates}`, ≈4-8 per ten minutes); quotas are applied only by
    the final selector. Chunk overlap is adaptive (20-45 s) instead of
    `chunk_seconds/8`.
  - The episode digest is multi-channel (an even sample plus sentences with
    numbers, questions, exclamations), the overview adds `context_limits`, and
    `episode_context.json` is cached under a key of transcript digest, prompt,
    model and language.
  - `analysis_metrics.json`: per-step counts and survival rates, exact-quote
    and low-confidence rates, boundary shift mean/p95, duplicate rate, topic
    diversity, quota fill rate, per-stage calls/latency/prompt size, and the
    retry budget spent.
- **llama.cpp provider.** One pooled `aiohttp` session per provider instead of
  one per request; exponential backoff with jitter between retries (a 503 gets
  a longer floor instead of a fixed 30 s); 4xx answers other than 408/429 are
  no longer retried; a rejected schema steps down to the stage's simplified
  schema before the permissive one. Malformed-JSON re-asks share an
  episode-wide budget (`analysis.json_retry_budget`).
- **Audio features** are probed only for the best `max_candidates`, on
  `parallelism` threads, and cached in `audio_features_cache.json`.
- **Burned subtitles are rendered in one encode.** The `.ass` is built for the
  clip's padded interval before the cut, instead of cutting the clip and then
  re-encoding it; the clean `reel_XX.nosubs.mp4` copy is now opt-in
  (`subtitles.keep_nosubs`, `--keep-nosubs`).
- Boundary snapping picks the cheapest nearby speech edge (sentence edges
  preferred, inward moves allowed) and never trims into the quote.

### Removed
- Orphaned experiment scripts `burn_drawtext_subs.py`, `burn_subs_pillow.py`
  (it imported Pillow, which is not a dependency) and `render_viral_subs.py`;
  the legacy benchmarks `scripts/test_models.py` and `test_models_v2.py`,
  which carried their own outdated prompts; the unused `refine`, `metadata`
  and `select_*` prompt templates.

### Fixed
- **Proofreading no longer desynchronises word timings.** Only
  `segments[].text` was corrected while `words` kept Whisper's raw tokens, so
  karaoke subtitles of every corrected sentence fell back to proportional
  timing, and quotes were verified against text the model never saw. Corrected
  text now gets its own word list (aligned token by token, timings carried
  over; the originals stay in `raw_words`).
- **`processing.quality_filters` are enforced at selection**, not only at the
  cut: every selected slot now goes to a clip that will actually be cut,
  instead of the output silently shrinking below its target. Filtered
  candidates are listed in `rejected_candidates.json`, and a clip type that can
  never pass the duration filters is reported in the log.
- A finished analysis that found nothing worth cutting is marked complete
  (`analysis_complete.json`) instead of being redone on every run.
- An interrupted yt-dlp merge left `… [id].f137.mp4` (video, no audio) and
  `… [id].f140.m4a` behind; both were taken for finished downloads and for
  separate episodes, and the silent one aborted every following run. Such
  pieces are now ignored, so yt-dlp resumes the merge.
- The Whisper OOM ladder (smaller batch, then CPU) never fired on a real OOM:
  faster-whisper decodes lazily and the generator was drained outside the
  retry loop. The API-mismatch fallback also silently dropped word timings
  and VAD; it now keeps every parameter.
- With YouTube unreachable the whole run aborted; it now logs the outage and
  processes what is already downloaded.
- A llama-server that failed to start was still polled for up to 300 s per
  episode.
- Transcripts and their SRT are written atomically.
- `crop_confidence` was filled with the duration-fit score; the field is now
  called `duration_fit_score`, and a stale `crop_confidence` is dropped.
- A cached episode overview was reused after the transcript, model or prompt
  changed in the same output folder.
- **llama-server's host-side prompt cache is now sized from free memory**
  (`cache_ram_mb: auto`). llama-server allows it 8192 MiB regardless of how much
  the machine has, and that cache — not the model weights — grew to ~6 GB of
  anonymous host memory and became this pipeline's share of a host OOM: with
  `n_gpu_layers: 999` the model sits entirely in VRAM while the host held 32
  context checkpoints of ~160 MB each. `auto` takes what is free at startup minus
  `cache_ram_reserve_mb`, capped by `cache_ram_max_mb`, so a run that shut a 12 GB
  VM down spends that memory instead of leaving it idle, while a busy host gets a
  small cache rather than an OOM. A plain number or `null` still work. Both this
  and `ctx_checkpoints` are passed only when the installed server understands
  them, since an unknown option makes llama-server exit rather than start.
- **llama-server's output is no longer discarded.** It went to `/dev/null`, which
  threw away the one report showing how many layers actually reached the GPU and
  which buffers stayed in host RAM. That hid a real problem: with
  `n_gpu_layers: 999` the server still held ~6 GB of *anonymous* host memory —
  CPU-side weights, not reclaimable page cache — and there was no way to see it.
  Output now appends to `llama_cpp.service.log_file` (`llama-server.log`);
  set it empty to go back to discarding.
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
