"""RU: Метаданные YouTube в обзоре эпизода, вычитке и статье.

EN: YouTube metadata in the episode overview, proofreading and the article.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from podcast_reels_forge import pipeline
from podcast_reels_forge.sources.episode_metadata import load_episode_metadata
from podcast_reels_forge.stages.analyze_stage import format_chapters, format_metadata_for_digest
from podcast_reels_forge.stages.proofread_stage import render_glossary

INFO = {
    "title": "Иван Петров о школе будущего",
    "channel": "Подкаст Прогресс",
    "description": "В гостях Иван Петров, основатель проекта Кванториум. Говорим про ЕГЭ и Сколково.",
    "tags": ["образование", "Кванториум"],
    "chapters": [
        {"start_time": 0.0, "title": "Вступление"},
        {"start_time": 754.0, "title": "Почему ЕГЭ устарел"},
    ],
}


def _write_info(source: Path) -> None:
    source.with_name(source.stem + ".info.json").write_text(
        json.dumps(INFO, ensure_ascii=False), encoding="utf-8",
    )


def test_metadata_is_read_next_to_the_source(tmp_path: Path) -> None:
    source = tmp_path / "2026-01-01 - Show [abc123def45].m4a"
    source.write_text("a")
    _write_info(source)

    meta = load_episode_metadata(source)

    assert meta is not None
    assert meta.title == INFO["title"]
    assert meta.chapters == [(0.0, "Вступление"), (754.0, "Почему ЕГЭ устарел")]
    assert load_episode_metadata(tmp_path / "other.mp4") is None


def test_glossary_keeps_names_and_skips_sentence_starts(tmp_path: Path) -> None:
    source = tmp_path / "ep.mp4"
    _write_info(source)
    glossary = load_episode_metadata(source).glossary()  # type: ignore[union-attr]

    assert "Иван Петров" in glossary
    assert "Кванториум" in glossary
    assert "Сколково" in glossary
    assert "Говорим" not in glossary, "a capital that only starts a sentence"


def test_overview_preamble_and_chapters() -> None:
    meta = {"title": "T", "description": "D", "chapters": [{"start": 754.0, "title": "ЕГЭ"}]}
    assert format_chapters(meta) == "Главы эпизода / Chapters: 12:34 ЕГЭ"
    preamble = format_metadata_for_digest(meta)
    assert "Title: T" in preamble and "Description: D" in preamble and "12:34" in preamble
    assert format_metadata_for_digest(None) == ""


def test_glossary_renders_into_the_prompt_or_vanishes() -> None:
    template = "A\n{glossary}\n\nСегменты (JSON):"
    assert "Иван Петров" in render_glossary(template, ["Иван Петров"], lang="ru")
    assert "{glossary}" not in render_glossary(template, [], lang="ru")


def test_pipeline_passes_metadata_on(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from types import SimpleNamespace

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    source = input_dir / "ep.mp4"
    source.write_text("v")
    _write_info(source)
    seen: dict[str, Any] = {}

    def fake_run(cmd: Any, **_: Any) -> SimpleNamespace:
        for arg in list(cmd):
            if Path(str(arg)).suffix in {".mp3", ".wav"}:
                Path(str(arg)).write_text("a")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def transcribe(config: pipeline.TranscribeConfig) -> Path:
        out = config.outdir / config.input_path.with_suffix(".json").name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"segments": [{"start": 0, "end": 5, "text": "hi"}]}))
        out.with_suffix(".srt").write_text("1\n")
        return out

    async def proofread(*, transcript_path: Path, output_path: Path, glossary: Any = None, **_: Any) -> Path:
        seen["glossary"] = glossary
        output_path.write_text(transcript_path.read_text())
        output_path.with_suffix(".srt").write_text("1\n")
        return output_path

    async def article(*, title: str, **_: Any) -> Path:
        seen["title"] = title
        return tmp_path / "a.md"

    async def analysis(*, outdir: Path, episode_metadata: Any = None, **_: Any) -> list[Any]:
        seen["metadata"] = episode_metadata
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "moments.json").write_text("[]")
        (outdir / "reels.md").write_text("#\n")
        return []

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    monkeypatch.setattr(pipeline, "ffmpeg_bin", lambda: "ffmpeg")
    monkeypatch.setattr(pipeline, "transcribe_file", transcribe)
    monkeypatch.setattr(pipeline, "run_proofread", proofread)
    monkeypatch.setattr(pipeline, "run_article", article)
    monkeypatch.setattr(pipeline, "run_staged_analysis", analysis)
    for name in ("llama_cpp_start", "llama_cpp_stop", "wait_for_server_ready", "_kill_llama_server"):
        monkeypatch.setattr(pipeline, name, lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "is_tcp_open", lambda *a: False)

    pipeline.run_pipeline(
        conf={
            "paths": {"input_dir": str(input_dir), "output_dir": str(tmp_path / "out")},
            "llama_cpp": {
                "url": "http://127.0.0.1:8080/completion",
                "service": {"auto_start": False},
                "roles": {"scout": "gemma4", "cleanup_refine": "gemma4", "judge_metadata": "gemma4"},
            },
            "proofread": {"enabled": True},
            "article": {"enabled": True},
            "subtitles": {"enabled": False},
        },
        repo_dir=tmp_path,
        quiet=True,
        verbose=False,
        stages={"transcribe", "proofread", "article", "analyze"},
    )

    assert "Иван Петров" in seen["glossary"]
    assert seen["title"] == INFO["title"]
    assert seen["metadata"]["chapters"][1] == {"start": 754.0, "title": "Почему ЕГЭ устарел"}
