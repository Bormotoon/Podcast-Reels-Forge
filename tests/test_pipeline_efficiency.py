"""RU: Эффективность прогона: порядок стадий, отпечатки, вычитка клипов.

EN: Run efficiency: stage ordering, fingerprints, clip-scoped proofreading.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from podcast_reels_forge import pipeline


def _conf(input_dir: Path, output_dir: Path, **extra: Any) -> dict[str, Any]:
    conf: dict[str, Any] = {
        "paths": {"input_dir": str(input_dir), "output_dir": str(output_dir)},
        "transcription": {"language": "ru", "device": "cpu"},
        "llama_cpp": {
            "roles": {"scout": "gemma4", "cleanup_refine": "gemma4", "judge_metadata": "gemma4"},
            "url": "http://127.0.0.1:8080/completion",
            "service": {"auto_start": True, "model_path": "/models/m.gguf"},
        },
        "processing": {"reels_count": 1, "reel_padding": 5},
        "video": {"threads": 1},
        "exports": {},
        "subtitles": {"enabled": False},
        "diarization": {"enabled": False},
        "prompts": {"language": "ru", "variant": "default"},
    }
    conf.update(extra)
    return conf


class _Env:
    """Records what the pipeline did, in order."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.proofread_ranges: list[Any] = []
        self.moments: list[dict[str, Any]] = [
            {"start": 100.0, "end": 140.0, "title": "T", "quote": "a b c", "score": 8},
        ]


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> _Env:
    record = _Env()

    def fake_run(cmd: Any, **_: Any) -> SimpleNamespace:
        for arg in list(cmd):
            if Path(str(arg)).suffix in {".mp3", ".wav"}:
                Path(str(arg)).write_text("audio")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    monkeypatch.setattr(pipeline, "ffmpeg_bin", lambda: "ffmpeg")
    monkeypatch.setattr(pipeline, "wait_for_server_ready", lambda *a, **kw: True)
    monkeypatch.setattr(pipeline, "_kill_llama_server", lambda *a: None)
    monkeypatch.setattr(pipeline, "is_tcp_open", lambda *a: False)

    def start(**_: Any) -> object:
        record.events.append("llama:start")
        return object()

    monkeypatch.setattr(pipeline, "llama_cpp_start", start)
    monkeypatch.setattr(pipeline, "llama_cpp_stop", lambda proc: record.events.append("llama:stop"))

    @contextlib.contextmanager
    def session() -> Iterator[None]:
        record.events.append("whisper:load")
        yield
        record.events.append("whisper:release")

    monkeypatch.setattr(pipeline, "whisper_model_session", session)

    def transcribe(config: pipeline.TranscribeConfig) -> Path:
        record.events.append(f"transcribe:{config.input_path.stem}")
        out = config.outdir / config.input_path.with_suffix(".json").name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"segments": [{"start": 0, "end": 200, "text": "hi"}]}))
        out.with_suffix(".srt").write_text("1\n")
        return out

    monkeypatch.setattr(pipeline, "transcribe_file", transcribe)

    async def analysis(*, transcript_path: Path, outdir: Path, **_: Any) -> list[Any]:
        record.events.append(f"analyze:{outdir.parent.name}")
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "moments.json").write_text(json.dumps(record.moments))
        (outdir / "reels.md").write_text("# r\n")
        return list(record.moments)

    monkeypatch.setattr(pipeline, "run_staged_analysis", analysis)

    async def proofread(*, transcript_path: Path, output_path: Path, time_ranges: Any = None, **_: Any) -> Path:
        record.events.append(f"proofread:{output_path.parent.name}")
        record.proofread_ranges.append(time_ranges)
        output_path.write_text(transcript_path.read_text())
        output_path.with_suffix(".srt").write_text("1\n")
        return output_path

    monkeypatch.setattr(pipeline, "run_proofread", proofread)

    def run_module(module: str, args: list[str], **_: Any) -> None:
        episode = Path(args[args.index("--input") + 1]).stem
        record.events.append(f"cut:{episode}")
        reels = Path(args[args.index("--outdir") + 1]) / "reels"
        reels.mkdir(parents=True, exist_ok=True)
        (reels / "reel_01.mp4").write_text("v")
        if "--transcript-json" in args:
            record.events.append("subs:" + Path(args[args.index("--transcript-json") + 1]).name)

    monkeypatch.setattr(pipeline, "run_module", run_module)
    monkeypatch.setattr(pipeline, "sync_reel_markdowns", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "sync_reel_burned_subtitles", lambda *a, **k: [])
    return record


def _episodes(tmp_path: Path, *stems: str) -> Path:
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    for stem in stems:
        (input_dir / f"{stem}.mp4").write_text("v")
    return input_dir


def _run(conf: dict[str, Any], tmp_path: Path, **kwargs: Any) -> Any:
    return pipeline.run_pipeline(conf=conf, repo_dir=tmp_path, quiet=True, verbose=False, **kwargs)


def test_stage_major_loads_each_model_once(env: _Env, tmp_path: Path) -> None:
    input_dir = _episodes(tmp_path, "a", "b")
    _run(_conf(input_dir, tmp_path / "out"), tmp_path)

    assert env.events == [
        "whisper:load", "transcribe:a", "transcribe:b", "whisper:release",
        "llama:start", "analyze:a", "analyze:b", "llama:stop",
        "cut:a", "cut:b",
    ]


def test_episode_scheduling_keeps_the_old_order(env: _Env, tmp_path: Path) -> None:
    input_dir = _episodes(tmp_path, "a", "b")
    conf = _conf(input_dir, tmp_path / "out", autonomy={"scheduling": "episode"})
    _run(conf, tmp_path)

    assert env.events.count("llama:start") == 2
    assert env.events.index("cut:a") < env.events.index("transcribe:b")


def test_no_llm_stage_means_no_llama_server(env: _Env, tmp_path: Path) -> None:
    input_dir = _episodes(tmp_path, "a")
    _run(_conf(input_dir, tmp_path / "out"), tmp_path, stages={"transcribe", "cut"})
    assert "llama:start" not in env.events


def test_analysis_reruns_only_when_its_inputs_change(env: _Env, tmp_path: Path) -> None:
    input_dir = _episodes(tmp_path, "a")
    conf = _conf(input_dir, tmp_path / "out")
    _run(conf, tmp_path)
    _run(conf, tmp_path)
    assert env.events.count("analyze:a") == 1, "unchanged inputs: cached"

    conf["processing"]["clips_per_hour"] = 12
    _run(conf, tmp_path)
    assert env.events.count("analyze:a") == 2, "changed config: redone"


def test_outputs_from_before_fingerprints_are_adopted(env: _Env, tmp_path: Path) -> None:
    input_dir = _episodes(tmp_path, "a")
    conf = _conf(input_dir, tmp_path / "out")
    _run(conf, tmp_path)
    (tmp_path / "out" / "a" / ".forge_state.json").unlink()

    _run(conf, tmp_path)

    assert env.events.count("analyze:a") == 1
    assert env.events.count("cut:a") == 1
    assert (tmp_path / "out" / "a" / ".forge_state.json").exists()


def test_new_moments_recut_the_reels(env: _Env, tmp_path: Path) -> None:
    input_dir = _episodes(tmp_path, "a")
    conf = _conf(input_dir, tmp_path / "out")
    _run(conf, tmp_path)
    stale = tmp_path / "out" / "a" / "gemma4" / "reels" / "reel_09.mp4"
    stale.write_text("old")

    env.moments = [{"start": 10.0, "end": 50.0, "title": "N", "quote": "x y z", "score": 9}]
    conf["processing"]["clips_per_hour"] = 12  # forces a new analysis
    _run(conf, tmp_path)

    assert env.events.count("cut:a") == 2
    assert not stale.exists(), "reels of the old moments are discarded"


def test_clip_scoped_proofread_runs_after_analysis(env: _Env, tmp_path: Path) -> None:
    input_dir = _episodes(tmp_path, "a")
    conf = _conf(
        input_dir,
        tmp_path / "out",
        proofread={"enabled": True, "scope": "clips"},
        subtitles={"enabled": True},
    )
    _run(conf, tmp_path)

    order = [e for e in env.events if e.split(":")[0] in {"analyze", "proofread", "cut", "subs"}]
    assert order == ["analyze:a", "proofread:a", "cut:a", "subs:a.proofread.json"]
    # 100..140 with 5 s padding and a 2 s margin.
    assert env.proofread_ranges == [[(93.0, 147.0)]]


def test_clip_scope_falls_back_to_full_with_the_article(env: _Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def article(**_: Any) -> Path:
        return tmp_path / "x.md"

    monkeypatch.setattr(pipeline, "run_article", article)
    input_dir = _episodes(tmp_path, "a")
    conf = _conf(
        input_dir,
        tmp_path / "out",
        proofread={"enabled": True, "scope": "clips"},
        article={"enabled": True},
    )
    _run(conf, tmp_path)

    assert env.events.index("proofread:a") < env.events.index("analyze:a")
    assert env.proofread_ranges == [None]


def test_listening_copy_is_optional(env: _Env, tmp_path: Path) -> None:
    input_dir = _episodes(tmp_path, "a")
    _run(_conf(input_dir, tmp_path / "out", audio={"listening_copy": False}), tmp_path, stages={"transcribe"})
    assert (input_dir / "a.wav").exists()
    assert not (input_dir / "a.mp3").exists()
