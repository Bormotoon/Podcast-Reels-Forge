"""RU: Контракт промптов и подготовка контекста для стадий.

EN: Prompt contract and the context prepared for each stage.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_reels_forge.analysis.contracts import coerce_moment_record
from podcast_reels_forge.analysis.transcript_index import TranscriptIndex
from podcast_reels_forge.stages.analyze_stage import (
    _load_prompt,
    _render_prompt,
    build_judge_payload,
    build_transcript_digest,
    format_episode_context,
)

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"

# Placeholders each stage's prompt is rendered with.
STAGE_PLACEHOLDERS = {
    "chunk": {"requirements", "chunk_json", "transcript", "episode_context"},
    "cleanup": {"requirements", "candidates_json"},
    "judge": {"requirements", "candidates_json", "episode_context"},
    "context": {"transcript_digest"},
}


@pytest.mark.parametrize("lang", ["ru", "en"])
@pytest.mark.parametrize("stage", sorted(STAGE_PLACEHOLDERS))
def test_default_prompts_exist_for_both_languages(stage: str, lang: str) -> None:
    assert (PROMPTS / lang / f"{stage}_default.txt").exists()


@pytest.mark.parametrize("lang", ["ru", "en"])
@pytest.mark.parametrize("stage", sorted(STAGE_PLACEHOLDERS))
def test_prompt_placeholders_are_all_supplied(stage: str, lang: str) -> None:
    """Every {placeholder} a prompt uses must be one the stage fills in.

    An unfilled placeholder would reach the model verbatim; the response
    schema in these prompts also uses braces, so this guards the difference.
    """
    text = _load_prompt(lang=lang, variant="default", name=stage)
    known = STAGE_PLACEHOLDERS[stage]
    rendered = _render_prompt(text, {key: "" for key in known})
    for placeholder in known:
        assert "{" + placeholder + "}" not in rendered


@pytest.mark.parametrize("lang", ["ru", "en"])
def test_scout_prompt_demands_verbatim_quotes(lang: str) -> None:
    """Quote verification only means something if the prompt asks for verbatim."""
    text = _load_prompt(lang=lang, variant="default", name="chunk").lower()
    assert "дословн" in text or "verbatim" in text


@pytest.mark.parametrize("lang", ["ru", "en"])
def test_scoring_prompts_anchor_the_score_scale(lang: str) -> None:
    """A bare 1-10 scale is uncalibrated; the anchors make it reproducible."""
    for stage in ("chunk", "judge"):
        text = _load_prompt(lang=lang, variant="default", name=stage)
        assert "1-10" in text
        for anchor in ("3:", "5:", "7:", "9:"):
            assert anchor in text, f"{lang}/{stage} is missing the {anchor} anchor"


def _index(sentence_count: int = 60) -> TranscriptIndex:
    sentences = []
    segments = []
    for i in range(sentence_count):
        start = i * 30.0
        text = f"Предложение {i} про школу и детей."
        sentences.append({"start": start, "end": start + 30.0, "text": text})
        words = []
        cursor = start
        for word in text.split():
            words.append({"start": cursor, "end": cursor + 1.0, "word": word})
            cursor += 1.0
        segments.append({"start": start, "end": start + 30.0, "text": text, "words": words})
    return TranscriptIndex.from_transcript({"sentences": sentences, "segments": segments})


def _record(start: float, end: float, title: str = "Момент"):
    record = coerce_moment_record(
        {
            "start": start,
            "end": end,
            "title": title,
            "quote": "Цитата",
            "why": "Причина",
            "score": 8.0,
        },
    )
    assert record is not None
    return record


def test_digest_samples_the_whole_episode_within_budget() -> None:
    index = _index()
    digest = build_transcript_digest(index, max_chars=500)
    assert digest
    assert len(digest) <= 501  # the ellipsis may add one char
    # Sampled across the episode rather than just the opening.
    assert "Предложение 0" in digest


def test_digest_is_empty_without_sentences() -> None:
    assert build_transcript_digest(TranscriptIndex.from_transcript({})) == ""


def test_judge_payload_carries_real_clip_edges() -> None:
    """The judge is told to grade openings, so give it the actual words."""
    index = _index()
    payload = build_judge_payload([_record(60.0, 150.0)], index)
    assert payload[0]["excerpt_head"]
    assert payload[0]["excerpt_tail"]
    assert payload[0]["excerpt_head"] != payload[0]["excerpt_tail"]


def test_judge_payload_caps_candidates_and_size() -> None:
    """The whole prompt has to fit inside ctx_size=8192."""
    index = _index()
    records = [_record(i * 100.0, i * 100.0 + 60.0, f"Момент {i}") for i in range(40)]
    payload = build_judge_payload(records, index, max_candidates=14)

    assert len(payload) == 14
    rendered = json.dumps(payload, ensure_ascii=False)
    # ~2-3 chars per token in Russian, so stay well under the context budget.
    assert len(rendered) < 11_000


def test_judge_payload_without_an_index_omits_excerpts() -> None:
    payload = build_judge_payload([_record(60.0, 150.0)], TranscriptIndex.from_transcript({}))
    assert "excerpt_head" not in payload[0]
    assert payload[0]["title"] == "Момент"


def test_episode_context_renders_only_what_it_has() -> None:
    rendered = format_episode_context({"summary": "Разговор о школе.", "topics": ["школа"]})
    assert "Разговор о школе." in rendered
    assert "школа" in rendered
    assert format_episode_context({}) == ""
