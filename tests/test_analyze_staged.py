"""RU: Сквозные тесты staged-анализа на фейковом LLM-провайдере.

EN: End-to-end tests of the staged analysis flow driven by a fake LLM
provider, so the whole pipeline is exercised without a llama.cpp server.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Callable

import pytest

from podcast_reels_forge.config import LlamaCppRoleMapping
from podcast_reels_forge.llm.providers import LlamaCppConfig, build_completion_payload
from podcast_reels_forge.llm.schemas import ANY_OBJECT_SCHEMA, MOMENTS_JSON_SCHEMA
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

    async def generate(self, prompt: str, *, temperature: float, timeout: int) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        return self._responder(prompt, self.calls)


def _moments_json(*moments: dict[str, Any]) -> str:
    return json.dumps({"moments": list(moments)}, ensure_ascii=False)


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


def _moment_in_chunk(prompt: str, title: str, *, offset: float = 5.0, length: float = 45.0) -> dict[str, Any]:
    """A candidate placed inside the chunk the prompt describes."""

    start, end = chunk_window(prompt)
    clip_start = min(start + offset, max(start, end - length))
    return _moment(clip_start, min(clip_start + length, end), title)


def prompt_candidates(prompt: str) -> list[dict[str, Any]]:
    """Read the candidate list a cleanup/judge prompt was rendered with."""

    # The prompt also contains a response-schema example, so anchor on the
    # candidates section rather than the first JSON-looking span.
    _head, separator, tail = prompt.rpartition("# Кандидаты")
    assert separator, "cleanup/judge prompt should carry a candidates section"
    parsed, _rest = json.JSONDecoder().raw_decode(tail.strip())
    assert isinstance(parsed, list)
    return parsed


def _refine_first(prompt: str, title: str, score: float = 8.0) -> str:
    """Echo back the first input candidate, retitled and re-rated.

    Real cleanup/judge stages filter and re-rate what they were given, and the
    stage now drops output that overlaps none of its input, so the fakes have
    to stay anchored to the candidates in their prompt.
    """

    first = prompt_candidates(prompt)[0]
    return _moments_json(_moment(first["start"], first["end"], title, score))


def _moment(start: float, end: float, title: str, score: float = 8.0) -> dict[str, Any]:
    return {
        "start": start,
        "end": end,
        "clip_type": "reel",
        "title": title,
        "quote": f"Яркая цитата про {title}",
        "why": "Понятная причина, почему это сработает в коротком видео",
        "score": score,
        "hook": f"Короткий хук про {title}",
    }


def _write_transcript(path: Path, *, duration: float = 2400.0) -> Path:
    """A timing_version-2 transcript long enough to produce several chunks."""

    segments = []
    sentences = []
    step = 30.0
    for index in range(int(duration // step)):
        start = index * step
        end = start + step
        text = f"Предложение номер {index} про школы, детей и результаты."
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
    outdir.mkdir()

    default_cleanup = cleanup or (lambda p, _i: _refine_first(p, "Чистый"))
    default_judge = judge or (lambda p, _i: _refine_first(p, "Финальный", 9.0))

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

    def fake_make_provider(_provider_name: str, *, model: str, **_kwargs: Any) -> FakeProvider:
        return providers[model]

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
        return _refine_first(prompt, "После ретрая", 9.0)

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
        moments = [_moment_in_chunk(prompt, f"Уникальный {call_index}", offset=120.0)]
        if not repeated:
            repeated["start"] = end - 60.0
            repeated["end"] = end - 15.0
        if start <= repeated["start"] and repeated["end"] <= end:
            moments.append(_moment(repeated["start"], repeated["end"], "Повторяющийся"))
        return _moments_json(*moments)

    _moments, providers, outdir = _run_analysis(monkeypatch, tmp_path, scout=scout)

    scouted = json.loads((outdir / "scout_candidates.json").read_text(encoding="utf-8"))
    repeated_scouted = [m for m in scouted if m["title"] == "Повторяющийся"]
    assert len(repeated_scouted) > 1, "the fixture must actually produce duplicates"

    # The cleanup stage sees the repeat exactly once, and every unique moment.
    cleanup_prompt = providers["cleanup_refine"].prompts[0]
    assert cleanup_prompt.count('"title": "Повторяющийся"') == 1
    unique_titles = {m["title"] for m in scouted if m["title"].startswith("Уникальный")}
    for title in unique_titles:
        assert f'"title": "{title}"' in cleanup_prompt


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
        return _refine_first(prompt, "Финальный", 9.0)

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
