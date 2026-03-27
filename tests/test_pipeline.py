"""Integration-level tests for the pipeline entrypoints."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

from podcast_reels_forge import pipeline

if TYPE_CHECKING:
    from pathlib import Path
    import pytest

    MonkeyPatch = pytest.MonkeyPatch


def test_pick_input_file_picks_newest(tmp_path: Path) -> None:
    """Ensure the newest media file is selected."""
    d = tmp_path / "input"
    d.mkdir()
    a = d / "a.mp4"
    b = d / "b.mp4"
    a.write_text("a")
    b.write_text("b")

    # RU: Делаем b «новее».
    # EN: Make b newer.
    b.touch()

    picked = pipeline.pick_input_file(d, (".mp4",))
    if picked is None:
        message = "Expected a picked file"
        raise AssertionError(message)
    if picked.name != "b.mp4":
        message = "Expected newest file 'b.mp4'"
        raise AssertionError(message)


def test_find_input_queue_creates_mp3_companion(
    monkeypatch: MonkeyPatch, tmp_path: Path,
) -> None:
    """Ensure missing same-stem MP3 is created from video input."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    video_path = input_dir / "episode.mp4"
    video_path.write_text("video")

    calls: list[list[str]] = []

    def fake_run(
        cmd: list[str] | tuple[str, ...],
        *,
        capture_output: bool = False,
        text: bool = False,
        **_: object,
    ) -> SimpleNamespace:
        cmd_list = list(cmd)
        calls.append(cmd_list)
        out_path = Path(cmd_list[-1])
        out_path.write_text("mp3")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)

    queue = pipeline.find_input_queue(input_dir)

    assert len(queue) == 1
    assert queue[0]["video"] == video_path
    assert queue[0]["audio"] == input_dir / "episode.mp3"
    assert (input_dir / "episode.mp3").exists()
    assert calls
    assert calls[0][0] == "ffmpeg"
    assert "-b:a" in calls[0]
    assert "320k" in calls[0]


def test_run_pipeline_builds_and_calls_stages(
    monkeypatch: MonkeyPatch, tmp_path: Path,
) -> None:
    """Ensure pipeline wires stages with correct arguments."""
    repo_dir = tmp_path

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "video.mp4").write_text("x")

    def fake_ffmpeg_run(
        cmd: list[str] | tuple[str, ...],
        *,
        capture_output: bool = False,
        text: bool = False,
        **_: object,
    ) -> SimpleNamespace:
        cmd_list = list(cmd)
        out_path = Path(cmd_list[-1])
        out_path.write_text("mp3")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pipeline.subprocess, "run", fake_ffmpeg_run)

    calls: list[tuple[str, list[str], dict[str, str] | None]] = []

    def fake_run_module(
        module: str,
        args: list[str],
        *,
        quiet: bool,
        verbose: bool,
        env: dict[str, str] | None = None,
    ) -> None:
        calls.append((module, list(args), env))

    monkeypatch.setattr(pipeline, "run_module", fake_run_module)

    started: list[tuple[str, int]] = []
    stopped: list[int] = []

    class FakeProc:
        def __init__(self, pid: int):
            self.pid = pid

    def fake_ollama_start(*, host: str, port: int):
        started.append((host, port))
        return FakeProc(123)

    def fake_ollama_stop(proc):
        stopped.append(getattr(proc, "pid", 0))

    monkeypatch.setattr(pipeline, "ollama_start", fake_ollama_start)
    monkeypatch.setattr(pipeline, "ollama_stop", fake_ollama_stop)

    conf = {
        "paths": {"input_dir": str(input_dir), "output_dir": str(tmp_path / "output")},
        "transcription": {"language": "auto"},
        "ollama": {
            "models": [
                "qwen3:latest",
                "deepseek-r1:8b",
                "gemma3:4b",
                "gemma2:9b",
                "gemini-3-flash-preview:latest",
            ],
            "url": "http://127.0.0.1:11434/api/generate",
        },
        "processing": {
            "reels_count": 2,
            "reel_min_duration": 10,
            "reel_max_duration": 20,
            "reel_padding": 5,
        },
        "video": {"threads": 1, "vertical_crop": True},
        "exports": {"webm": True, "gif": False, "audio_only": True},
        "subtitles": {
            "enabled": True,
            "font": "assets/fonts/custom.ttf",
            "css": "assets/subtitles/custom.css",
            "wrap_words": False,
        },
        "diarization": {"enabled": False},
        "prompts": {"language": "auto", "variant": "a"},
    }

    pipeline.run_pipeline(conf=conf, repo_dir=repo_dir, quiet=True, verbose=False)

    module_names = [c[0] for c in calls]
    expected_modules = {
        "podcast_reels_forge.scripts.transcribe",
        "podcast_reels_forge.scripts.analyze",
        "podcast_reels_forge.scripts.video_processor",
    }
    if not expected_modules.issubset(module_names):
        message = "Expected all stage modules to be invoked"
        raise AssertionError(message)

    analyze_calls = [c for c in calls if "podcast_reels_forge.scripts.analyze" in c[0]]
    if len(analyze_calls) != 5:
        message = f"Expected 5 analyze runs (one per model), got: {len(analyze_calls)}"
        raise AssertionError(message)
    for _module, analyze_args, analyze_env in analyze_calls:
        if "--provider" not in analyze_args or "--prompt-variant" not in analyze_args:
            message = "Analyze stage missing required CLI args"
            raise AssertionError(message)
        if not analyze_env or analyze_env.get("FORGE_MANAGED_OLLAMA") != "1":
            message = "Expected pipeline to pass FORGE_MANAGED_OLLAMA=1 to analyze stage"
            raise AssertionError(message)

    if started != [("127.0.0.1", 11434)]:
        message = f"Expected Ollama to be started once, got: {started}"
        raise AssertionError(message)
    if stopped != [123]:
        message = f"Expected Ollama to be stopped, got: {stopped}"
        raise AssertionError(message)

    video_calls = [c for c in calls if "podcast_reels_forge.scripts.video_processor" in c[0]]
    if len(video_calls) != 5:
        message = f"Expected 5 video runs (one per model), got: {len(video_calls)}"
        raise AssertionError(message)
    for _module, video_args, _env in video_calls:
        if "--export-webm" not in video_args or "--export-audio" not in video_args:
            message = "Video stage missing export flags"
            raise AssertionError(message)
        if "--padding" not in video_args:
            message = "Video stage missing padding flag"
            raise AssertionError(message)
        if "--burn-subtitles" not in video_args or "--transcript-json" not in video_args:
            message = "Video stage missing subtitle args"
            raise AssertionError(message)
        if "--no-subtitle-wrap-words" not in video_args:
            message = "Video stage missing wrap-word toggle"
            raise AssertionError(message)
        if str(tmp_path / "assets" / "fonts" / "custom.ttf") not in video_args:
            message = "Video stage missing configured subtitle font path"
            raise AssertionError(message)
        if str(tmp_path / "assets" / "subtitles" / "custom.css") not in video_args:
            message = "Video stage missing configured subtitle css path"
            raise AssertionError(message)


def test_run_pipeline_syncs_reel_markdowns_for_existing_outputs(
    monkeypatch: MonkeyPatch, tmp_path: Path,
) -> None:
    """Ensure cached outputs still get per-reel markdown files."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "video.mp4").write_text("x")

    def fake_ffmpeg_run(
        cmd: list[str] | tuple[str, ...],
        *,
        capture_output: bool = False,
        text: bool = False,
        **_: object,
    ) -> SimpleNamespace:
        cmd_list = list(cmd)
        out_path = Path(cmd_list[-1])
        out_path.write_text("mp3")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pipeline.subprocess, "run", fake_ffmpeg_run)

    output_root = tmp_path / "output"
    model_dir = output_root / "video" / "qwen3"
    reels_dir = model_dir / "reels"
    rejected_dir = reels_dir / "rejected"
    rejected_dir.mkdir(parents=True)

    transcript_path = output_root / "video" / "video.json"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(
        json.dumps({"segments": [], "duration": 0.0}),
        encoding="utf-8",
    )
    (output_root / "video" / "video.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n")

    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "moments.json").write_text(
        json.dumps(
            [
                {
                    "start": 10.0,
                    "end": 20.0,
                    "title": "First",
                    "quote": "Quote 1",
                    "why": "Why 1",
                    "score": 9,
                    "hook": "Hook 1",
                    "caption": "Caption 1",
                    "hashtags": ["#podcast"],
                },
                {
                    "start": 30.0,
                    "end": 40.0,
                    "title": "Second",
                    "quote": "Quote 2",
                    "why": "Why 2",
                    "score": 8,
                    "hook": "Hook 2",
                    "caption": "Caption 2",
                    "hashtags": ["#reels"],
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (model_dir / "reels.md").write_text("# Reels Suggestions\n\n(existing)\n", encoding="utf-8")
    (reels_dir / "reel_01.mp4").write_text("mp4")
    (rejected_dir / "reel_02.mp4").write_text("mp4")

    calls: list[str] = []

    def fake_run_module(
        module: str,
        args: list[str],
        *,
        quiet: bool,
        verbose: bool,
        env: dict[str, str] | None = None,
    ) -> None:
        calls.append(module)

    monkeypatch.setattr(pipeline, "run_module", fake_run_module)

    started: list[tuple[str, int]] = []
    stopped: list[int] = []

    class FakeProc:
        def __init__(self, pid: int):
            self.pid = pid

    monkeypatch.setattr(
        pipeline,
        "ollama_start",
        lambda *, host, port: (started.append((host, port)) or FakeProc(321)),
    )
    monkeypatch.setattr(
        pipeline,
        "ollama_stop",
        lambda proc: stopped.append(getattr(proc, "pid", 0)),
    )
    monkeypatch.setattr(pipeline, "get_ollama_models", lambda url: ["qwen3:latest"])
    monkeypatch.setattr(pipeline, "pull_ollama_model", lambda url, model: True)

    subtitle_sync_calls: list[tuple[Path, Path]] = []

    def fake_sync_reel_burned_subtitles(
        moments: list[dict[str, object]],
        reels_root: Path,
        *,
        transcript_json_path: Path,
        padding: float,
        settings: object,
        verbose: bool = False,
    ) -> list[Path]:
        subtitle_sync_calls.append((reels_root, transcript_json_path))
        return []

    monkeypatch.setattr(
        pipeline,
        "sync_reel_burned_subtitles",
        fake_sync_reel_burned_subtitles,
    )

    conf = {
        "paths": {"input_dir": str(input_dir), "output_dir": str(output_root)},
        "transcription": {"language": "auto"},
        "ollama": {
            "models": ["qwen3:latest"],
            "url": "http://127.0.0.1:11434/api/generate",
        },
        "processing": {
            "reels_count": 1,
            "reel_min_duration": 10,
            "reel_max_duration": 20,
            "reel_padding": 5,
        },
        "video": {"threads": 1, "vertical_crop": True},
        "exports": {"webm": False, "gif": False, "audio_only": False},
        "subtitles": {"enabled": True},
        "diarization": {"enabled": False},
        "prompts": {"language": "auto", "variant": "default"},
    }

    pipeline.run_pipeline(conf=conf, repo_dir=tmp_path, quiet=True, verbose=False)

    assert calls == []
    assert started == [("127.0.0.1", 11434)]
    assert stopped == [321]
    assert (reels_dir / "reel_01.md").exists()
    assert (rejected_dir / "reel_02.md").exists()
    assert subtitle_sync_calls == [(reels_dir, transcript_path)]
