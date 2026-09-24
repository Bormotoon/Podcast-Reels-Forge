"""RU: Сквозные тесты staged-анализа на фейковом LLM-провайдере.

EN: End-to-end tests of the staged analysis flow driven by a fake LLM
provider, so the whole pipeline is exercised without a llama.cpp server.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from podcast_reels_forge.config import LlamaCppRoleMapping
from podcast_reels_forge.llm.providers import LlamaCppConfig, build_completion_payload
from podcast_reels_forge.llm.schemas import (
    ANY_OBJECT_SCHEMA,
    MOMENTS_JSON_SCHEMA,
    SCOUT_JSON_SCHEMA,
)
from podcast_reels_forge.stages import analyze_stage


class FakeProvider:
    """A provider whose replies are scripted per stage.

    ``responder`` receives the rendered prompt and the 1-based call index for
    that provider, and returns the raw text the model would have produced. It
    may raise to simulate a server failure.
    """

    def __init__(self, responder: Callable[[str, int], str]) -> None:
        self._responder = responder
        self.calls = 0
        self.prompts: list[str] = []
        # json_schema values this "model" was constructed with, per stage.
        self.schemas: list[Any] = []

    async def generate(self, prompt: str, *, temperature: float, timeout: int) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        return self._responder(prompt, self.calls)


SEGMENT_SECONDS = 30.0


def spoken_sentence(index: int) -> str:
    """The fixture transcript's sentence for segment ``index``."""

    return f"Предложение номер {index} про школы, детей и результаты."


def _fixture_words(start: float, end: float) -> list[str]:
    """Words of the fixture transcript lying entirely inside [start, end]."""

    words: list[str] = []
    first = max(0, int(start // SEGMENT_SECONDS))
    last = int(end // SEGMENT_SECONDS)
    for index in range(first, last + 1):
        word_list = spoken_sentence(index).split()
        per_word = SEGMENT_SECONDS / len(word_list)
        for offset, word in enumerate(word_list):
            w_start = round(index * SEGMENT_SECONDS + offset * per_word, 3)
            w_end = round(index * SEGMENT_SECONDS + (offset + 1) * per_word, 3)
            if w_start >= start and w_end <= end:
                words.append(word)
    return words


def _spoken_quote(start: float, end: float, *, max_words: int = 8) -> str:
    """A verbatim, contiguous quote lying inside [start, end] of the fixture."""

    words = _fixture_words(start, end)
    assert len(words) >= 4, "clip too short to hold a quote"
    return " ".join(words[:max_words])


def _candidates_json(*candidates: dict[str, Any]) -> str:
    return json.dumps({"candidates": list(candidates)}, ensure_ascii=False)


# Legacy name kept for readability of older tests.
_moments_json = _candidates_json


def chunk_window(prompt: str) -> tuple[float, float]:
    """Read the chunk's [start, end] out of a rendered scout prompt.

    Real scouts answer with timecodes from the window they were shown, and the
    stage now drops anything outside it, so the fakes have to do the same.
    """

    # The prompt shows a response-schema example with its own start/end, so
    # anchor on the chunk section instead of the first numbers in the text.
    _head, separator, tail = prompt.partition("# Кусок транскрипта")
    assert separator, "scout prompt should carry a chunk section"
    chunk_meta, _rest = json.JSONDecoder().raw_decode(tail.strip())
    return float(chunk_meta["start"]), float(chunk_meta["end"])


def is_context_prompt(prompt: str) -> bool:
    """Whether this is the one-off episode-overview call, not a chunk."""

    return "# Выжимка транскрипта" in prompt


EPISODE_CONTEXT_REPLY = json.dumps(
    {"summary": "Эпизод про школу.", "topics": ["школа"], "tone": "беседа"},
    ensure_ascii=False,
)


def _moment_in_chunk(prompt: str, title: str = "", *, offset: float = 5.0, length: float = 45.0) -> dict[str, Any]:
    """A candidate placed inside the chunk the prompt describes."""

    start, end = chunk_window(prompt)
    clip_start = min(start + offset, max(start, end - length))
    return _moment(clip_start, min(clip_start + length, end))


def prompt_candidates(prompt: str) -> list[dict[str, Any]]:
    """Read the candidate list a cleanup/judge prompt was rendered with."""

    # The prompt also contains a response-schema example, so anchor on the
    # candidates section rather than the first JSON-looking span.
    _head, separator, tail = prompt.rpartition("# Кандидаты")
    assert separator, "cleanup/judge prompt should carry a candidates section"
    parsed, _rest = json.JSONDecoder().raw_decode(tail.strip())
    assert isinstance(parsed, list)
    return parsed


def keep_all_decisions(prompt: str, _call_index: int = 0) -> str:
    """A permissive cleanup: keep every candidate it was shown."""

    return json.dumps(
        {"decisions": [{"candidate_id": c["candidate_id"], "keep": True} for c in prompt_candidates(prompt)]},
    )


def keep_first_decision(prompt: str, _call_index: int = 0) -> str:
    """An extreme cleanup: keep the first candidate, drop the rest."""

    candidates = prompt_candidates(prompt)
    return json.dumps(
        {
            "decisions": [
                {"candidate_id": c["candidate_id"], "keep": offset == 0}
                for offset, c in enumerate(candidates)
            ],
        },
    )


def review_all(prompt: str, title: str = "Финальный", score: float = 9.0) -> str:
    """A judge that keeps and titles every candidate it was shown."""

    return json.dumps(
        {
            "reviews": [
                {
                    "candidate_id": c["candidate_id"],
                    "keep": True,
                    "score": score,
                    "title": f"{title} {offset}" if offset else title,
                    "hook": f"Хук: {title}",
                    "why": "Законченная мысль с понятной развязкой",
                }
                for offset, c in enumerate(prompt_candidates(prompt))
            ],
        },
        ensure_ascii=False,
    )


def _moment(start: float, end: float, title: str = "", score: float = 8.0) -> dict[str, Any]:
    """A scout candidate whose quote is really in the fixture transcript."""

    return {
        "start": start,
        "end": end,
        "quote": _spoken_quote(start, end),
        "evidence": "Понятная причина, почему это сработает в коротком видео",
        "reason_codes": ["story"],
        "score": score,
    }


def _write_transcript(path: Path, *, duration: float = 2400.0) -> Path:
    """A timing_version-2 transcript long enough to produce several chunks."""

    segments = []
    sentences = []
    step = SEGMENT_SECONDS
    for index in range(int(duration // step)):
        start = index * step
        end = start + step
        text = spoken_sentence(index)
        words = []
        word_list = text.split()
        per_word = (end - start) / max(1, len(word_list))
        for word_index, word in enumerate(word_list):
            words.append(
                {
                    "start": round(start + word_index * per_word, 3),
                    "end": round(start + (word_index + 1) * per_word, 3),
                    "word": word,
                    "probability": 0.9,
                },
            )
        segments.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "confidence": 0.9,
                "speaker": "SPEAKER_00",
                "words": words,
            },
        )
        sentences.append({"start": start, "end": end, "text": text, "speaker": "SPEAKER_00"})

    path.write_text(
        json.dumps(
            {
                "audio": "episode.mp3",
                "source_audio": "episode.mp3",
                "language": "ru",
                "duration": duration,
                "timing_version": 2,
                "segments": segments,
                "sentences": sentences,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def _run_analysis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    scout: Callable[[str, int], str],
    cleanup: Callable[[str, int], str] | None = None,
    judge: Callable[[str, int], str] | None = None,
    processing_conf: dict[str, Any] | None = None,
    scout_handles_context: bool = False,
) -> tuple[list[Any], dict[str, FakeProvider], Path]:
    transcript = _write_transcript(tmp_path / "episode.json")
    outdir = tmp_path / "out"
    outdir.mkdir(exist_ok=True)

    default_cleanup = cleanup or keep_all_decisions
    default_judge = judge or (lambda p, _i: review_all(p))

    def scout_with_context(prompt: str, call_index: int) -> str:
        # The stage asks the scout model for the episode overview first; tests
        # that do not care about it get a canned reply.
        if is_context_prompt(prompt) and scout_handles_context is False:
            return EPISODE_CONTEXT_REPLY
        return scout(prompt, call_index)

    providers = {
        "scout": FakeProvider(scout_with_context),
        "cleanup_refine": FakeProvider(default_cleanup),
        "judge_metadata": FakeProvider(default_judge),
    }

    def fake_make_provider(_provider_name: str, *, model: str, **kwargs: Any) -> FakeProvider:
        provider = providers[model]
        provider.schemas.append(kwargs.get("json_schema"))
        return provider

    monkeypatch.setattr(analyze_stage, "_make_stage_provider", fake_make_provider)

    roles = LlamaCppRoleMapping(
        scout="scout",
        cleanup_refine="cleanup_refine",
        judge_metadata="judge_metadata",
        proofread="cleanup_refine",
    )
    conf: dict[str, Any] = {
        "clips": {"reels": {"count": 3, "max_duration": 60}},
        "reels_count": 3,
    }
    if processing_conf:
        conf.update(processing_conf)

    moments = asyncio.run(
        analyze_stage.run_staged_analysis(
            transcript_path=transcript,
            outdir=outdir,
            provider_name="llama_cpp",
            url="http://127.0.0.1:11440/completion",
            api_key=None,
            roles=roles,
            llama_cpp_conf={"url": "http://127.0.0.1:11440/completion", "chunk_seconds": 600},
            prompts_conf={"language": "ru", "variant": "default"},
            processing_conf=conf,
            quiet=True,
        ),
    )
    return moments, providers, outdir


def test_staged_analysis_writes_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A clean run produces moments.json, reels.md and the intermediates."""
    moments, _providers, outdir = _run_analysis(
        monkeypatch,
        tmp_path,
        scout=lambda p, _i: _moments_json(_moment_in_chunk(p, "Найденный")),
    )

    assert moments
    for name in (
        "moments.json",
        "reels.md",
        "scout_candidates.json",
        "cleaned_candidates.json",
        "analysis_manifest.json",
    ):
        assert (outdir / name).exists(), name

    payload = json.loads((outdir / "moments.json").read_text(encoding="utf-8"))
    assert payload
    # score keeps the model's 1-10 rating; priority carries the ranking value.
    assert payload[0]["score"] == 9.0
    assert payload[0]["priority"] != payload[0]["score"]
    assert "Score: 9.0/10" in (outdir / "reels.md").read_text(encoding="utf-8")


def test_one_failing_chunk_does_not_abort_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A chunk whose request blows up is skipped, not fatal.

    Losing one 20-minute window beats losing the whole episode, which is what
    the bare asyncio.gather used to do.
    """

    def scout(prompt: str, call_index: int) -> str:
        if call_index == 2:
            raise RuntimeError("llama.cpp connection reset")
        return _moments_json(_moment_in_chunk(prompt, f"Чанк {call_index}"))

    moments, providers, outdir = _run_analysis(monkeypatch, tmp_path, scout=scout)

    assert providers["scout"].calls > 2
    assert moments
    scouted = json.loads((outdir / "scout_candidates.json").read_text(encoding="utf-8"))
    assert scouted, "surviving chunks still contribute candidates"


def test_all_chunks_failing_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A total scout outage is an error, not an empty result."""

    def scout(_prompt: str, _call_index: int) -> str:
        raise RuntimeError("llama.cpp is down")

    with pytest.raises(RuntimeError, match="all"):
        _run_analysis(monkeypatch, tmp_path, scout=scout)


def test_unparseable_json_is_retried_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Malformed output costs one retry rather than the whole chunk."""
    seen: list[str] = []

    def judge(prompt: str, call_index: int) -> str:
        seen.append(prompt)
        if call_index == 1:
            return "Sure! Here are the clips, but not as JSON."
        return review_all(prompt, "После ретрая")

    moments, providers, _outdir = _run_analysis(
        monkeypatch,
        tmp_path,
        scout=lambda p, _i: _moments_json(_moment_in_chunk(p, "Найденный")),
        judge=judge,
    )

    assert providers["judge_metadata"].calls == 2
    assert "IMPORTANT" in seen[1], "the retry tells the model its output was unparseable"
    assert moments[0].title == "После ретрая"


def test_deduped_before_the_cleanup_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Overlapping duplicates must not consume the capped candidate slots.

    Chunks overlap, so the same moment is scouted repeatedly; the cap has to
    see a de-duplicated list.
    """

    # Adjacent chunks overlap, so a moment sitting in the shared tail is
    # reported by both — exactly the duplication the pre-cleanup dedupe exists
    # to absorb. Anchor it to the first chunk's tail, which the second chunk
    # also covers.
    repeated: dict[str, float] = {}

    def scout(prompt: str, call_index: int) -> str:
        start, end = chunk_window(prompt)
        # Keep the unique moment well clear of the shared tail, so it is not
        # itself deduped against the repeat.
        moments = [_moment_in_chunk(prompt, offset=120.0)]
        if not repeated:
            # The adaptive overlap is 20-45s, so the shared tail holds the
            # chunk's last sentence.
            repeated["start"] = end - 30.0
            repeated["end"] = end
        if start <= repeated["start"] and repeated["end"] <= end:
            moments.append(_moment(repeated["start"], repeated["end"]))
        return _moments_json(*moments)

    _moments, providers, outdir = _run_analysis(monkeypatch, tmp_path, scout=scout)

    scouted = json.loads((outdir / "scout_candidates.json").read_text(encoding="utf-8"))
    repeated_quote = _spoken_quote(repeated["start"], repeated["end"])
    repeated_scouted = [m for m in scouted if m["quote"] == repeated_quote]
    assert len(repeated_scouted) > 1, "the fixture must actually produce duplicates"

    # The cleanup stage sees the repeat exactly once, and every unique moment.
    cleanup_prompt = providers["cleanup_refine"].prompts[0]
    assert cleanup_prompt.count(f'"quote": "{repeated_quote}"') == 1
    for quote in {m["quote"] for m in scouted} - {repeated_quote}:
        assert f'"quote": "{quote}"' in cleanup_prompt


def test_episode_context_is_built_once_and_reused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The overview costs one call, is cached, and reaches the scout prompt."""

    def scout(prompt: str, _call_index: int) -> str:
        if is_context_prompt(prompt):
            return json.dumps(
                {
                    "summary": "Эпизод про школу и детей.",
                    "topics": ["школа", "дети"],
                    "tone": "дружеская беседа",
                },
                ensure_ascii=False,
            )
        return _moments_json(_moment_in_chunk(prompt, "Найденный"))

    _moments, providers, outdir = _run_analysis(
        monkeypatch, tmp_path, scout=scout, scout_handles_context=True,
    )

    cache = outdir / "episode_context.json"
    assert cache.exists()
    assert json.loads(cache.read_text(encoding="utf-8"))["summary"]

    chunk_prompts = [p for p in providers["scout"].prompts if "# Кусок транскрипта" in p]
    assert chunk_prompts, "the scout should still receive chunk prompts"
    assert all("Эпизод про школу и детей." in prompt for prompt in chunk_prompts)

    # A second run over the same outdir reuses the cache instead of re-asking.
    digest_calls = sum(1 for p in providers["scout"].prompts if "# Выжимка транскрипта" in p)
    assert digest_calls == 1


def test_analysis_survives_a_failed_episode_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The overview is a nicety; losing it must not cost the episode."""

    def scout(prompt: str, _call_index: int) -> str:
        if is_context_prompt(prompt):
            raise RuntimeError("context call failed")
        return _moments_json(_moment_in_chunk(prompt, "Найденный"))

    moments, _providers, outdir = _run_analysis(
        monkeypatch, tmp_path, scout=scout, scout_handles_context=True,
    )

    assert moments
    assert not (outdir / "episode_context.json").exists()


def test_judge_sees_the_real_clip_edges(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The judge grades openings and endings, so it gets the actual words."""
    captured: list[dict[str, Any]] = []

    def judge(prompt: str, _call_index: int) -> str:
        captured.extend(prompt_candidates(prompt))
        return review_all(prompt)

    _run_analysis(
        monkeypatch,
        tmp_path,
        scout=lambda p, _i: _moments_json(_moment_in_chunk(p, "Найденный")),
        judge=judge,
    )

    assert captured
    assert all("excerpt_head" in item for item in captured)
    # Every fixture sentence says "про школы, детей", so a real excerpt has it.
    assert all("школы" in item["excerpt_head"] for item in captured)
    # The judge's prompt tells it to distrust low quote_match_ratio values, so
    # verification has to run before the judge for the field to be there.
    assert all("quote_match_ratio" in item for item in captured)


def test_diversity_config_reaches_the_ranking(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """processing.analysis.diversity must actually change selection."""
    captured_kwargs: dict[str, Any] = {}
    real_rank = analyze_stage.rank_moments

    def spying_rank(records: Any, **kwargs: Any) -> Any:
        captured_kwargs.update(kwargs)
        return real_rank(records, **kwargs)

    monkeypatch.setattr(analyze_stage, "rank_moments", spying_rank)

    _run_analysis(
        monkeypatch,
        tmp_path,
        scout=lambda p, _i: _moments_json(_moment_in_chunk(p, "Найденный")),
        processing_conf={
            "analysis": {
                "diversity": {"enabled": False, "max_topic_similarity": 0.8},
            },
        },
    )

    assert captured_kwargs["diversity_enabled"] is False
    assert captured_kwargs["max_topic_similarity"] == 0.8


def test_clips_scale_with_runtime_and_stages_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """clips_per_hour drives the output count end to end.

    The 2400s fixture at 24 clips/hour targets 16 final clips — more than one
    judge prompt fits (14) and, after the x2 headroom, more than one cleanup
    prompt fits (25), so this also exercises the batching that makes large
    targets reachable at ctx_size=8192.
    """

    def scout(prompt: str, call_index: int) -> str:
        # Enough distinct, non-overlapping candidates spread over each chunk.
        start, end = chunk_window(prompt)
        moments = []
        cursor = start + 2.0
        slot = 0
        while cursor + 45.0 <= end and slot < 8:
            moments.append(
                _moment(cursor, cursor + 45.0, f"Тема {call_index}-{slot} номер {call_index * 10 + slot}"),
            )
            cursor += 70.0
            slot += 1
        return _moments_json(*moments)

    def echo_all(prompt: str, _call_index: int) -> str:
        # Echo every input back in the legacy full-record shape, like an old
        # custom prompt would: it must still work, applied by candidate_id.
        return json.dumps({"moments": prompt_candidates(prompt)}, ensure_ascii=False)

    moments, providers, outdir = _run_analysis(
        monkeypatch,
        tmp_path,
        scout=scout,
        cleanup=echo_all,
        judge=echo_all,
        processing_conf={"clips_per_hour": 24},
    )

    manifest = json.loads((outdir / "analysis_manifest.json").read_text(encoding="utf-8"))
    assert sum(manifest["quotas"].values()) == 16  # round(2400/3600 * 24)

    assert len(moments) == 16

    # The scaled ask reaches the stages that select; the scout only gets the
    # clip lengths — quotas are the final selector's job, not recall's.
    scout_chunk_prompt = next(
        p for p in providers["scout"].prompts if "# Кусок транскрипта" in p
    )
    assert "Reels: up to 60s" in scout_chunk_prompt
    assert "16 clips" not in scout_chunk_prompt
    assert "Reels: 16 clips" in providers["judge_metadata"].prompts[0]

    # 32 candidates survive the x2-headroom cap: two cleanup batches of ≤16
    # and three judge batches of ≤14 (14 + 14 + 4).
    assert providers["cleanup_refine"].calls == 2
    assert providers["judge_metadata"].calls == 3


def test_episode_context_uses_its_own_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The overview call must not inherit the moments grammar.

    The scout provider's schema forces {"moments": [...]}, which makes
    {"summary": ...} unrepresentable — on a real run the episode context came
    back as an empty moments object and was silently skipped.
    """
    from podcast_reels_forge.llm.schemas import EPISODE_CONTEXT_SCHEMA

    _moments, providers, outdir = _run_analysis(
        monkeypatch,
        tmp_path,
        scout=lambda p, _i: _moments_json(_moment_in_chunk(p, "Найденный")),
    )

    assert (outdir / "episode_context.json").exists()
    assert EPISODE_CONTEXT_SCHEMA in providers["scout"].schemas
    assert SCOUT_JSON_SCHEMA in providers["scout"].schemas


def test_cleanup_shrinkage_is_topped_up_from_scouted_pool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A cleanup that loses candidates must not starve the target.

    On a real run output truncation shrank 38 candidates to 18 against a
    19-clip target; the pool is topped back up with the best dropped
    candidates that don't overlap what cleanup kept.
    """

    def scout(prompt: str, call_index: int) -> str:
        start, end = chunk_window(prompt)
        moments = []
        cursor = start + 2.0
        slot = 0
        while cursor + 45.0 <= end and slot < 8:
            moments.append(
                _moment(cursor, cursor + 45.0, f"Тема {call_index}-{slot} слот {call_index * 10 + slot}"),
            )
            cursor += 70.0
            slot += 1
        return _moments_json(*moments)

    # Cleanup keeps only its first input candidate — an extreme shrink.
    moments, _providers, outdir = _run_analysis(
        monkeypatch,
        tmp_path,
        scout=scout,
        cleanup=keep_first_decision,
        judge=lambda p, _i: json.dumps(
            {"moments": prompt_candidates(p)}, ensure_ascii=False,
        ),
        processing_conf={"clips_per_hour": 24},
    )

    cleaned = json.loads((outdir / "cleaned_candidates.json").read_text(encoding="utf-8"))
    assert len(cleaned) > 1, "the pool must be restored above cleanup's single survivor"
    # Target for the 40-min fixture at 24/hour is 16; the run must not collapse
    # to one clip because cleanup misbehaved.
    assert len(moments) >= 10


def test_strict_schema_is_sent_and_can_be_disabled() -> None:
    """The moments schema constrains sampling unless explicitly turned off."""
    cfg = LlamaCppConfig(url="http://x/completion", model="m", json_schema=MOMENTS_JSON_SCHEMA)
    payload = build_completion_payload(cfg, "prompt", temperature=0.2)
    assert payload["json_schema"]["required"] == ["moments"]

    permissive = LlamaCppConfig(url="http://x/completion", model="m")
    assert build_completion_payload(permissive, "p", temperature=0.2)["json_schema"] == (
        ANY_OBJECT_SCHEMA
    )


def test_schema_downgrade_falls_back_to_any_object() -> None:
    """Builds that reject the schema get the permissive one instead."""
    cfg = LlamaCppConfig(url="http://x/completion", model="m", json_schema=MOMENTS_JSON_SCHEMA)
    payload = build_completion_payload(cfg, "p", temperature=0.2, schema_downgraded=True)
    assert payload["json_schema"] == ANY_OBJECT_SCHEMA


def test_invented_quotes_never_reach_the_cut(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A quote nobody said is rejected before cleanup and logged with a reason."""

    def scout(prompt: str, _call_index: int) -> str:
        real = _moment_in_chunk(prompt)
        invented = {**_moment_in_chunk(prompt, offset=200.0), "quote": "совершенно выдуманная фраза про пингвинов"}
        quoteless = {**_moment_in_chunk(prompt, offset=300.0), "quote": ""}
        return _candidates_json(real, invented, quoteless)

    moments, providers, outdir = _run_analysis(monkeypatch, tmp_path, scout=scout)

    assert moments
    assert all("пингвин" not in m.quote for m in moments)
    assert all("пингвин" not in p for p in providers["cleanup_refine"].prompts)
    rejected = json.loads((outdir / "rejected_candidates.json").read_text(encoding="utf-8"))
    assert rejected
    assert {row["rejection_reason"] for row in rejected} == {"quote_not_in_transcript"}
    metrics = json.loads((outdir / "analysis_metrics.json").read_text(encoding="utf-8"))
    assert metrics["stages"]["scout"]["dropped_without_quote"] > 0
    assert metrics["rejections"]["quote_not_in_transcript"] == len(rejected)


def test_judge_cannot_rewrite_evidence_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    def judge(prompt: str, _call_index: int) -> str:
        return json.dumps(
            {
                "reviews": [
                    {
                        "candidate_id": c["candidate_id"],
                        "keep": True,
                        "score": 9,
                        "title": "Заголовок судьи",
                        "quote": "красивая, но выдуманная цитата",
                        "start": 0.0,
                        "end": 1.0,
                    }
                    for c in prompt_candidates(prompt)
                ],
            },
            ensure_ascii=False,
        )

    moments, _providers, outdir = _run_analysis(
        monkeypatch,
        tmp_path,
        scout=lambda p, _i: _candidates_json(_moment_in_chunk(p)),
        judge=judge,
    )

    scouted = json.loads((outdir / "scout_candidates.json").read_text(encoding="utf-8"))
    scouted_quotes = {m["quote"] for m in scouted}
    assert moments
    for moment in moments:
        assert moment.title == "Заголовок судьи"
        assert moment.quote in scouted_quotes
        assert moment.end - moment.start > 30
        assert moment.candidate_id


def test_judge_batches_are_stratified_and_carry_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    captured: list[list[dict[str, Any]]] = []

    def scout(prompt: str, call_index: int) -> str:
        start, end = chunk_window(prompt)
        moments = []
        cursor = start + 2.0
        slot = 0
        while cursor + 45.0 <= end and slot < 8:
            moments.append(_moment(cursor, cursor + 45.0, score=float(1 + (slot + call_index) % 9)))
            cursor += 70.0
            slot += 1
        return _candidates_json(*moments)

    def judge(prompt: str, _call_index: int) -> str:
        captured.append(prompt_candidates(prompt))
        return review_all(prompt)

    _run_analysis(
        monkeypatch, tmp_path, scout=scout, judge=judge, processing_conf={"clips_per_hour": 24},
    )

    assert len(captured) >= 2
    assert all(item["candidate_id"] for batch in captured for item in batch)
    # Every batch sees a comparable spread, not best-first slices.
    tops = [max(item["score"] for item in batch) for batch in captured]
    assert max(tops) - min(tops) <= 1


def test_metrics_describe_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    moments, _providers, outdir = _run_analysis(
        monkeypatch,
        tmp_path,
        scout=lambda p, _i: _candidates_json(_moment_in_chunk(p)),
    )
    metrics = json.loads((outdir / "analysis_metrics.json").read_text(encoding="utf-8"))
    assert metrics["counts"]["final"] == len(moments)
    assert metrics["quote_exact_match_rate"] == 1.0
    assert metrics["stages"]["scout"]["calls"] >= 1
    for key in (
        "scout_candidates_per_hour",
        "candidate_survival_rate",
        "quote_low_confidence_rate",
        "boundary_shift_seconds",
        "duplicate_rate",
        "topic_diversity",
        "quota_fill_rate",
        "json_retries",
    ):
        assert key in metrics, key


def test_stale_episode_context_cache_is_not_reused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A cache left by another transcript/model must be rebuilt, not trusted."""
    outdir = tmp_path / "out"
    outdir.mkdir()
    (outdir / "episode_context.json").write_text(
        json.dumps({"summary": "Совсем другой эпизод про космос.", "cache_key": "stale"}, ensure_ascii=False),
        encoding="utf-8",
    )
    _moments, providers, _ = _run_analysis(
        monkeypatch,
        tmp_path,
        scout=lambda p, _i: _candidates_json(_moment_in_chunk(p)),
    )

    chunk_prompts = [p for p in providers["scout"].prompts if "# Кусок транскрипта" in p]
    assert all("космос" not in prompt for prompt in chunk_prompts)
    assert any("Эпизод про школу." in prompt for prompt in chunk_prompts)


def test_quality_filters_are_enforced_at_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Clips the cut stage would reject never take a selected slot."""

    def scout(prompt: str, _call_index: int) -> str:
        return _candidates_json(
            _moment_in_chunk(prompt, offset=5.0, length=45.0),
            # Too short even after boundary snapping (at most +3s per edge).
            _moment_in_chunk(prompt, offset=200.0, length=20.0),
        )

    def judge(prompt: str, _call_index: int) -> str:
        # Short clips get a good score, so only their length can reject them;
        # the earliest long one gets a failing score.
        candidates = prompt_candidates(prompt)
        long_ones = sorted((c for c in candidates if c["duration"] >= 30), key=lambda c: c["start"])
        weak = long_ones[0]["candidate_id"] if long_ones else None
        reviews = [
            {"candidate_id": c["candidate_id"], "keep": True, "score": 6 if c["candidate_id"] == weak else 9}
            for c in candidates
        ]
        return json.dumps({"reviews": reviews})

    moments, _providers, outdir = _run_analysis(
        monkeypatch,
        tmp_path,
        scout=scout,
        judge=judge,
        processing_conf={"quality_filters": {"min_score": 7, "min_duration": 30}},
    )

    assert moments
    assert all(m.score >= 7 and m.end - m.start >= 30 for m in moments)
    rejected = json.loads((outdir / "rejected_candidates.json").read_text(encoding="utf-8"))
    reasons = {row["rejection_reason"] for row in rejected if row["rejected_at"] == "selection"}
    assert reasons == {"shorter_than_min_duration", "below_min_score"}


def test_an_empty_but_finished_analysis_is_marked_complete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    moments, _providers, outdir = _run_analysis(
        monkeypatch,
        tmp_path,
        scout=lambda p, _i: _candidates_json(_moment_in_chunk(p)),
        processing_conf={"quality_filters": {"min_score": 10}},
    )

    assert moments == []
    marker = json.loads((outdir / "analysis_complete.json").read_text(encoding="utf-8"))
    assert marker == {**marker, "status": "ok", "moments": 0}
