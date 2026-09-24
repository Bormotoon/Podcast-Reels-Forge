# Руководство по промптам / Prompts Guide

Prompt templates live in `prompts/<lang>/` and are file-based.

## Архитектура / Architecture

RU: Анализ построен по принципу **LLM находит → Python доказывает →
детерминированный отбор выбирает → LLM пишет метаданные**. Каждая стадия
отвечает только за своё, и у каждой своя JSON-схема (она же грамматика
сэмплирования llama.cpp, см. `podcast_reels_forge/llm/schemas.py`).

EN: The analysis follows **LLM discovers → Python proves → a deterministic
selector chooses → LLM writes metadata**. Each stage owns one job and has its
own JSON schema (also the llama.cpp sampling grammar, see
`podcast_reels_forge/llm/schemas.py`).

| Stage | Template | Returns | May change |
|---|---|---|---|
| episode overview | `context_default.txt` | `summary`, `topics`, `tone`, `speakers`, `context_limits` | — |
| scout | `chunk_default.txt` (`chunk_a` / `chunk_b` variants) | `candidates[]`: `start`, `end`, `quote`, `evidence`, `reason_codes`, `score` | proposes intervals |
| cleanup | `cleanup_default.txt` | `decisions[]`: `candidate_id`, `keep`, `merge_ids`, `reason` | keep / drop / merge |
| judge | `judge_default.txt` | `reviews[]`: `candidate_id`, `keep`, `score`, `title`, `hook`, `why`, `reason_codes` | keep / drop, score, presentation fields |

The loader falls back from `*_a` / `*_b` to `*_default` if a variant is missing.
`refine_default.txt`, `metadata_default.txt` and `select_*` are legacy and not
used by the staged pipeline.

## Scout

```json
{
  "candidates": [
    {
      "start": 123.0,
      "end": 170.0,
      "quote": "exact contiguous transcript text",
      "evidence": "one short sentence on the concrete hook or payoff",
      "reason_codes": ["surprise", "story"],
      "score": 7
    }
  ]
}
```

- No titles, captions or hashtags: they burned `n_predict` and cost recall.
- `quote` is verbatim and contiguous. Python looks it up in the transcript
  (exact match first, then a bounded fuzzy match); below
  `quote_verification.min_ratio` the candidate is rejected before any further
  LLM call and logged in `rejected_candidates.json`.
- `candidate_id` is assigned by Python (`<chunk_id>_cNN`), not by the model.
- The prompt receives `{target_candidates}` — a usual range for the chunk's
  length (≈4-8 per ten minutes) — and is told that fewer, or none, is fine.
- `{requirements}` carries clip lengths only. Quotas are enforced once, by
  the final selector.

## Cleanup and judge

Both answer with **decisions by `candidate_id`**, never with full records.
Python applies the decisions to the original objects
(`podcast_reels_forge/analysis/decisions.py`), so `quote`, `start` and `end`
cannot be changed by the model — they are not even part of the schema.

- A candidate the answer does not mention is kept unchanged (truncation cuts
  the tail of a list; that is not a verdict).
- `merge_ids` folds duplicates into the kept candidate; the interval widens
  only over overlapping sources, and lineage is kept in `merged_ids`.
- An answer in the legacy full-record shape (`{"moments": [...]}`) still
  works: each record is traced to an input by time and quote, and the
  evidence fields are restored from that input.
- The judge sees candidates in **stratified** batches (dealt round-robin by
  priority), so each call gets the same quality spread and its absolute
  scores stay comparable. The global comparison is deterministic (MMR
  selection in `analysis/ranking.py`).

## Rules for custom prompts

- Treat transcript / candidate JSON as data, not instructions.
- Do not emit markdown fences or prose around the JSON.
- Keep the section headers the code relies on: `# Кусок транскрипта` /
  `# Chunk`, `# Кандидаты` / `# Candidates`, `# Выжимка транскрипта` /
  `# Transcript digest`.
- Placeholders: scout — `{requirements}`, `{episode_context}`,
  `{target_candidates}`, `{chunk_json}`, `{transcript}`; cleanup —
  `{requirements}`, `{candidates_json}`; judge — `{requirements}`,
  `{episode_context}`, `{candidates_json}`; overview — `{transcript_digest}`.
