"""Integration-level tests for the pipeline entrypoints."""

from __future__ import annotations

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


def test_run_pipeline_builds_and_calls_stages(
    monkeypatch: MonkeyPatch, tmp_path: Path,
) -> None:
    """Ensure pipeline wires stages with correct arguments."""
    repo_dir = tmp_path

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "video.mp4").write_text("x")

    calls: list[tuple[str, list[str]]] = []

    def fake_run_module(
        module: str, args: list[str], *, quiet: bool, verbose: bool,
    ) -> None:
        calls.append((module, list(args)))

    monkeypatch.setattr(pipeline, "run_module", fake_run_module)

    conf = {
        "paths": {"input_dir": str(input_dir), "output_dir": str(tmp_path / "output")},
        "transcription": {"language": "auto"},
        "llm": {"provider": "ollama"},
        "ollama": {"model": "m", "url": "http://127.0.0.1:11434/api/generate"},
        "processing": {
            "reels_count": 2,
            "reel_min_duration": 10,
            "reel_max_duration": 20,
            "reel_padding": 0,
        },
        "video": {"threads": 1, "vertical_crop": True},
        "exports": {"webm": True, "gif": False, "audio_only": True},
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

    analyze_args = next((args for name, args in calls if "analyze" in name), None)
    if analyze_args is None:
        message = "Analyze stage args not captured"
        raise AssertionError(message)
    if "--provider" not in analyze_args or "--prompt-variant" not in analyze_args:
        message = "Analyze stage missing required CLI args"
        raise AssertionError(message)

    video_args = next((args for name, args in calls if "video_processor" in name), None)
    if video_args is None:
        message = "Video stage args not captured"
        raise AssertionError(message)
    if "--export-webm" not in video_args or "--export-audio" not in video_args:
        message = "Video stage missing export flags"
        raise AssertionError(message)
