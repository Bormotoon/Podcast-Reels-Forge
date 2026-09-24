"""RU: Прогон без присмотра: изоляция эпизодов, отчёт, блокировка, preflight.

EN: Unattended runs: episode isolation, the report, the lock, preflight.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from podcast_reels_forge import pipeline
from podcast_reels_forge.autonomy import notify, setup_file_logging
from podcast_reels_forge.preflight import run_preflight
from podcast_reels_forge.run_report import (
    EXIT_FATAL,
    EXIT_OK,
    EXIT_PARTIAL,
    FAILED,
    RunReport,
)
from podcast_reels_forge.sources.youtube import YouTubeError
from podcast_reels_forge.stages.fetch_stage import find_local_copy
from podcast_reels_forge.utils.run_lock import RunLock, RunLockBusy


def _conf(input_dir: Path, output_dir: Path, **extra: Any) -> dict[str, Any]:
    conf: dict[str, Any] = {
        "paths": {"input_dir": str(input_dir), "output_dir": str(output_dir)},
        "transcription": {"language": "ru", "device": "cpu"},
        "llama_cpp": {
            "roles": {"scout": "gemma4", "cleanup_refine": "gemma4", "judge_metadata": "gemma4"},
            "url": "http://127.0.0.1:8080/completion",
            "service": {"auto_start": False},
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


def _stub_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: Any, **_: Any) -> SimpleNamespace:
        for arg in list(cmd):
            if Path(str(arg)).suffix in {".mp3", ".wav"}:
                Path(str(arg)).write_text("audio")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    monkeypatch.setattr(pipeline, "ffmpeg_bin", lambda: "ffmpeg")
    monkeypatch.setattr(pipeline, "llama_cpp_start", lambda **kw: None)
    monkeypatch.setattr(pipeline, "llama_cpp_stop", lambda proc: None)
    monkeypatch.setattr(pipeline, "wait_for_server_ready", lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, "_kill_llama_server", lambda *a: None)
    monkeypatch.setattr(pipeline, "is_tcp_open", lambda *a: False)


def _fake_transcribe(broken_stems: set[str]) -> Any:
    def transcribe(config: pipeline.TranscribeConfig) -> Path:
        if config.input_path.stem in broken_stems:
            raise RuntimeError("CUDA failed: out of memory")
        out = config.outdir / config.input_path.with_suffix(".json").name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"segments": [{"start": 0, "end": 5, "text": "hi"}]}))
        out.with_suffix(".srt").write_text("1\n")
        return out

    return transcribe


def _fake_analysis(calls: list[str]) -> Any:
    async def analysis(*, transcript_path: Path, outdir: Path, **_: Any) -> list[Any]:
        calls.append(transcript_path.stem)
        outdir.mkdir(parents=True, exist_ok=True)
        moments = [{"start": 1.0, "end": 40.0, "title": "T", "quote": "a b c", "score": 8}]
        (outdir / "moments.json").write_text(json.dumps(moments))
        (outdir / "reels.md").write_text("# r\n")
        return moments

    return analysis


# -- isolation ---------------------------------------------------------------


def test_one_broken_episode_does_not_stop_the_queue(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    for stem in ("a", "b", "c"):
        (input_dir / f"{stem}.mp4").write_text("x")
    _stub_environment(monkeypatch)
    analysed: list[str] = []
    monkeypatch.setattr(pipeline, "transcribe_file", _fake_transcribe({"b"}))
    monkeypatch.setattr(pipeline, "run_staged_analysis", _fake_analysis(analysed))

    def failing_cut(module: str, args: list[str], **_: Any) -> None:
        if "a.mp4" in " ".join(args):
            raise SystemExit(1)

    monkeypatch.setattr(pipeline, "run_module", failing_cut)

    report = pipeline.run_pipeline(
        conf=_conf(input_dir, tmp_path / "out"), repo_dir=tmp_path, quiet=True, verbose=False,
    )

    assert analysed == ["a", "c"], "b failed to transcribe; a and c still analysed"
    assert report.episodes["b"].stages["transcribe"].status == FAILED
    assert "out of memory" in report.episodes["b"].stages["transcribe"].detail
    assert "analyze" not in report.episodes["b"].stages
    assert report.episodes["a"].stages["cut"].status == FAILED
    assert report.episodes["c"].stages["cut"].status == "done"
    assert report.outcome == "partial"
    assert report.exit_code() == EXIT_PARTIAL


def test_a_file_without_audio_costs_only_its_episode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    (input_dir / "silent.mp4").write_text("x")
    (input_dir / "good.mp4").write_text("x")
    _stub_environment(monkeypatch)

    def companions(source: Path) -> tuple[Path, Path]:
        if source.stem == "silent":
            raise SystemExit("Failed to create audio companions for silent.mp4: no audio")
        mp3, wav = source.with_suffix(".mp3"), source.with_suffix(".wav")
        mp3.write_text("a")
        wav.write_text("a")
        return mp3, wav

    monkeypatch.setattr(pipeline, "_ensure_audio_companions", companions)
    monkeypatch.setattr(pipeline, "transcribe_file", _fake_transcribe(set()))

    report = pipeline.run_pipeline(
        conf=_conf(input_dir, tmp_path / "out"),
        repo_dir=tmp_path, quiet=True, verbose=False, stages={"transcribe"},
    )

    assert report.episodes["silent"].stages["audio"].status == FAILED
    assert report.episodes["good"].stages["transcribe"].status == "done"


def test_download_fragments_are_not_episodes(tmp_path: Path) -> None:
    """An interrupted yt-dlp merge leaves per-format pieces that must be ignored."""
    input_dir = tmp_path / "in"
    (input_dir / "youtube").mkdir(parents=True)
    (input_dir / "youtube" / "2026-01-01 - Show [abc123def45].f137.mp4").write_text("v")
    (input_dir / "youtube" / "2026-01-01 - Show [abc123def45].f140.m4a").write_text("a")
    (input_dir / "episode.mp4").write_text("v")

    queue = pipeline.find_input_queue(input_dir, ensure_companions=False)

    assert [item["stem"] for item in queue] == ["episode"]
    assert find_local_copy(input_dir / "youtube", "abc123def45") is None


def test_youtube_outage_falls_back_to_downloaded_episodes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    input_dir = tmp_path / "in"
    download_dir = input_dir / "youtube"
    download_dir.mkdir(parents=True)
    (download_dir / "2026-01-01 - Show [abc123def45].m4a").write_text("a")
    (input_dir / "local.mp4").write_text("v")
    _stub_environment(monkeypatch)

    def offline(*_a: Any, **_k: Any) -> Any:
        raise YouTubeError("network is unreachable")

    monkeypatch.setattr(pipeline, "resolve_sources", offline)
    transcribed: list[str] = []

    def transcribe(config: pipeline.TranscribeConfig) -> Path:
        transcribed.append(config.input_path.stem)
        return _fake_transcribe(set())(config)

    monkeypatch.setattr(pipeline, "transcribe_file", transcribe)
    conf = _conf(input_dir, tmp_path / "out", youtube={"download_dir": str(download_dir)})

    report = pipeline.run_pipeline(
        conf=conf, repo_dir=tmp_path, quiet=True, verbose=False,
        stages={"fetch", "transcribe"}, youtube_sources=["@show"],
    )

    assert transcribed == ["2026-01-01 - Show [abc123def45]"]
    assert report.outcome == "partial"
    assert any("YouTube" in event["message"] for event in report.events)


# -- report ------------------------------------------------------------------


def test_report_outcomes_and_exit_codes(tmp_path: Path) -> None:
    report = RunReport()
    report.record("ep", "transcribe", "done")
    assert report.exit_code() == EXIT_OK
    report.record("ep", "analyze", FAILED, detail="boom")
    assert report.exit_code() == EXIT_PARTIAL
    report.fatal("preflight: no ffmpeg")
    assert report.exit_code() == EXIT_FATAL

    report.finish()
    path = report.write(tmp_path / "_runs")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["outcome"] == "failed"
    assert data["episodes"]["ep"]["stages"]["analyze"]["detail"] == "boom"
    assert json.loads((tmp_path / "_runs" / "latest.json").read_text()) == data


def test_notify_command_gets_the_outcome(tmp_path: Path) -> None:
    marker = tmp_path / "notified"
    conf = {"autonomy": {"notify": {"when": "failure", "command": f'echo "$FORGE_OUTCOME $FORGE_EXIT_CODE" > {marker}'}}}
    ok = RunReport()
    notify(conf, ok, None)
    assert not marker.exists(), "on: failure stays quiet when all is well"

    failed = RunReport()
    failed.record("ep", "cut", FAILED)
    notify(conf, failed, tmp_path / "r.json")
    assert marker.read_text().strip() == "partial 3"


def test_file_log_captures_info_even_when_the_console_is_quiet(tmp_path: Path) -> None:
    import logging

    root = logging.getLogger()
    saved = (root.level, list(root.handlers))
    try:
        root.setLevel(logging.ERROR)
        path = setup_file_logging({"autonomy": {"log_dir": str(tmp_path)}}, tmp_path)
        assert path is not None
        logging.getLogger("Forge").info("hello from the night run")
        for handler in root.handlers:
            handler.flush()
        assert "hello from the night run" in path.read_text(encoding="utf-8")
    finally:
        for handler in list(root.handlers):
            if handler not in saved[1]:
                root.removeHandler(handler)
                handler.close()
        root.setLevel(saved[0])


# -- lock --------------------------------------------------------------------


def test_second_lock_holder_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "forge.lock"
    with RunLock(path):
        with pytest.raises(RunLockBusy, match=str(os.getpid())):
            RunLock(path).acquire()
    # Released: the next run may go.
    with RunLock(path):
        pass


# -- preflight ---------------------------------------------------------------


def test_preflight_catches_missing_llama_server_and_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from podcast_reels_forge import preflight as pf

    monkeypatch.setattr(pf, "ffmpeg_bin", lambda: "sh")
    monkeypatch.setattr(pf, "is_tcp_open", lambda *a: False)
    monkeypatch.delenv("PYANNOTE_TOKEN", raising=False)
    conf = _conf(tmp_path / "in", tmp_path / "out")
    conf["llama_cpp"]["service"] = {"auto_start": True, "model_path": str(tmp_path / "nope.gguf")}
    conf["diarization"] = {"enabled": True}
    conf["video"] = {"smart_crop_face": False}

    result = run_preflight(conf, stages=pipeline.PIPELINE_STAGES, repo_dir=tmp_path)

    joined = " ".join(result.errors)
    assert not result.ok
    assert "nope.gguf" in joined
    assert "PYANNOTE_TOKEN" in joined


def test_preflight_passes_when_the_server_is_already_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from podcast_reels_forge import preflight as pf

    monkeypatch.setattr(pf, "ffmpeg_bin", lambda: "sh")
    monkeypatch.setattr(pf, "is_tcp_open", lambda *a: True)
    conf = _conf(tmp_path / "in", tmp_path / "out", video={"smart_crop_face": False})

    result = run_preflight(conf, stages={"analyze", "cut"}, repo_dir=tmp_path)

    assert result.ok, result.errors


def test_preflight_refuses_a_full_disk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from podcast_reels_forge import preflight as pf

    monkeypatch.setattr(pf, "ffmpeg_bin", lambda: "sh")
    conf = _conf(tmp_path / "in", tmp_path / "out", autonomy={"min_free_disk_gb": 10**9})

    result = run_preflight(conf, stages={"cut"}, repo_dir=tmp_path)

    assert any("min_free_disk_gb" in error for error in result.errors)


def test_finished_empty_analysis_is_not_redone(tmp_path: Path) -> None:
    folder = tmp_path / "gemma4"
    folder.mkdir()
    moments, reels = folder / "moments.json", folder / "reels.md"
    moments.write_text("[]")
    reels.write_text("# r\n")
    # A crash placeholder: empty and unvouched — redo it.
    assert not pipeline._analysis_outputs_ready(moments, reels, validate_json=True)
    (folder / "analysis_complete.json").write_text(json.dumps({"status": "ok", "moments": 0}))
    assert pipeline._analysis_outputs_ready(moments, reels, validate_json=True)
