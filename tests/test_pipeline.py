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



def _write_ffmpeg_outputs(cmd: list[str]) -> None:
    """Create every output file an ffmpeg command names.

    The pipeline builds the MP3 and WAV companions in one pass, so a fake that
    only writes ``cmd[-1]`` silently loses the first output.
    """
    for arg in cmd:
        suffix = Path(arg).suffix.lower()
        if suffix in {".mp3", ".wav"}:
            Path(arg).write_text(suffix.lstrip("."))


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
        _write_ffmpeg_outputs(cmd_list)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    monkeypatch.setattr(pipeline, "ffmpeg_bin", lambda: "ffmpeg")

    queue = pipeline.find_input_queue(input_dir)

    assert len(queue) == 1
    assert queue[0]["video"] == video_path
    assert queue[0]["audio"] == input_dir / "episode.mp3"
    assert (input_dir / "episode.mp3").exists()
    assert calls
    assert str(calls[0][0]).endswith("ffmpeg")
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
        _write_ffmpeg_outputs(list(cmd))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pipeline.subprocess, "run", fake_ffmpeg_run)
    monkeypatch.setattr(pipeline, "ffmpeg_bin", lambda: "ffmpeg")

    transcribe_calls: list[pipeline.TranscribeConfig] = []

    def fake_transcribe_file(config: pipeline.TranscribeConfig) -> Path:
        transcribe_calls.append(config)
        out_path = config.outdir / config.input_path.with_suffix(".json").name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "language": config.language,
                    "duration": 120.0,
                    "segments": [
                        {"start": 0.0, "end": 10.0, "text": "hello world"},
                    ],
                },
            ),
            encoding="utf-8",
        )
        out_path.with_suffix(".srt").write_text(
            "1\n00:00:00,000 --> 00:00:10,000\nhello world\n",
            encoding="utf-8",
        )
        return out_path

    monkeypatch.setattr(pipeline, "transcribe_file", fake_transcribe_file)

    analysis_calls: list[dict[str, object]] = []

    async def fake_run_staged_analysis(
        *,
        transcript_path: Path,
        outdir: Path,
        provider_name: str,
        url: str,
        api_key: str | None,
        roles: object,
        llama_cpp_conf: dict[str, object],
        prompts_conf: dict[str, object],
        processing_conf: dict[str, object],
        diarization_path: Path | None = None,
        quiet: bool = False,
        verbose: bool = False,
        progress: bool = False,
    ) -> list[dict[str, object]]:
        analysis_calls.append(
            {
                "transcript_path": transcript_path,
                "outdir": outdir,
                "provider_name": provider_name,
                "url": url,
                "api_key": api_key,
                "roles": roles,
                "llama_cpp_conf": llama_cpp_conf,
                "prompts_conf": prompts_conf,
                "processing_conf": processing_conf,
                "diarization_path": diarization_path,
                "quiet": quiet,
                "verbose": verbose,
                "progress": progress,
            },
        )
        outdir.mkdir(parents=True, exist_ok=True)
        moments = [
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
        ]
        (outdir / "moments.json").write_text(
            json.dumps(moments, ensure_ascii=False),
            encoding="utf-8",
        )
        (outdir / "reels.md").write_text(
            "# Reels Suggestions\n\n## 1. First [reel]\n",
            encoding="utf-8",
        )
        return moments

    monkeypatch.setattr(pipeline, "run_staged_analysis", fake_run_staged_analysis)

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

    sync_markdown_calls: list[tuple[list[dict[str, object]], Path]] = []

    def fake_sync_reel_markdowns(
        moments: list[dict[str, object]],
        reels_root: Path,
        *,
        max_description_chars: int = 1000,
        hashtag_count: int = 5,
    ) -> list[Path]:
        sync_markdown_calls.append((moments, reels_root))
        return []

    monkeypatch.setattr(pipeline, "sync_reel_markdowns", fake_sync_reel_markdowns)

    started: list[tuple[str, int]] = []
    stopped: list[int] = []

    class FakeProc:
        def __init__(self, pid: int):
            self.pid = pid

    def fake_llama_cpp_start(*, host: str, port: int, service_conf: dict[str, object] | None = None):
        _ = service_conf
        started.append((host, port))
        return FakeProc(123)

    def fake_llama_cpp_stop(proc):
        stopped.append(getattr(proc, "pid", 0))

    monkeypatch.setattr(pipeline, "llama_cpp_start", fake_llama_cpp_start)
    monkeypatch.setattr(pipeline, "llama_cpp_stop", fake_llama_cpp_stop)
    monkeypatch.setattr(pipeline, "wait_for_server_ready", lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, "_kill_llama_server", lambda *a: None)
    monkeypatch.setattr(pipeline, "is_tcp_open", lambda *a: False)

    conf = {
        "paths": {"input_dir": str(input_dir), "output_dir": str(tmp_path / "output")},
        "transcription": {
            "language": "auto",
            "device": "cpu",
            "model": "large-v3",
            "compute_type": "float32",
        },
        "llama_cpp": {
            "roles": {
                "scout": "gemma4",
                "cleanup_refine": "gemma4",
                "judge_metadata": "gemma4",
            },
            "url": "http://127.0.0.1:8080/v1/chat/completions",
            "service": {
                "auto_start": True,
                "model_path": str(tmp_path / "model.gguf"),
            },
            "timeout": 240,
            "temperature": 0.2,
            "chunk_seconds": 900,
            "max_chars_chunk": 12000,
            "watchdog": {
                "enabled": True,
                "first_token_timeout": 60,
                "stall_timeout": 90,
                "log_interval": 10,
                "max_retries": 1,
            },
            "fallback_models": [],
            "role_overrides": {},
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
        "prompts": {"language": "auto", "variant": "default"},
    }

    pipeline.run_pipeline(conf=conf, repo_dir=repo_dir, quiet=True, verbose=False)

    assert len(transcribe_calls) == 1
    # The models read the WAV; the transcript still lands under the shared stem.
    assert transcribe_calls[0].input_path == input_dir / "video.wav"
    assert transcribe_calls[0].outdir == tmp_path / "output" / "video"

    assert len(analysis_calls) == 1
    roles = analysis_calls[0]["roles"]
    assert getattr(roles, "judge_metadata") == "gemma4"
    assert analysis_calls[0]["transcript_path"] == tmp_path / "output" / "video" / "video.json"
    assert analysis_calls[0]["outdir"] == tmp_path / "output" / "video" / "gemma4"

    if started != [("127.0.0.1", 8080)]:
        message = f"Expected llama.cpp to be started once, got: {started}"
        raise AssertionError(message)
    if stopped != [123]:
        message = f"Expected llama.cpp to be stopped, got: {stopped}"
        raise AssertionError(message)

    assert [c[0] for c in calls] == ["podcast_reels_forge.scripts.video_processor"]
    video_args = calls[0][1]
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

    assert len(sync_markdown_calls) == 1
    assert sync_markdown_calls[0][1] == tmp_path / "output" / "video" / "gemma4" / "reels"
    assert (tmp_path / "output" / "video" / "gemma4" / "moments.json").exists()


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
        _write_ffmpeg_outputs(list(cmd))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pipeline.subprocess, "run", fake_ffmpeg_run)
    monkeypatch.setattr(pipeline, "ffmpeg_bin", lambda: "ffmpeg")

    output_root = tmp_path / "output"
    model_dir = output_root / "video" / "gemma4"
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
        "llama_cpp_start",
        lambda *, host, port, service_conf=None: (started.append((host, port)) or FakeProc(321)),
    )
    monkeypatch.setattr(
        pipeline,
        "llama_cpp_stop",
        lambda proc: stopped.append(getattr(proc, "pid", 0)),
    )
    monkeypatch.setattr(pipeline, "wait_for_server_ready", lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, "_kill_llama_server", lambda *a: None)
    monkeypatch.setattr(pipeline, "is_tcp_open", lambda *a: False)

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
        "llama_cpp": {
            "roles": {
                "scout": "gemma4",
                "cleanup_refine": "gemma4",
                "judge_metadata": "gemma4",
            },
            "url": "http://127.0.0.1:8080/v1/chat/completions",
            "service": {
                "auto_start": True,
                "model_path": str(tmp_path / "model.gguf"),
            },
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
    assert started == [("127.0.0.1", 8080)]
    assert stopped == [321]
    assert (reels_dir / "reel_01.md").exists()
    assert (rejected_dir / "reel_02.md").exists()
    assert subtitle_sync_calls == [(reels_dir, transcript_path)]


def test_run_pipeline_routes_proofread_transcript_to_analysis(
    monkeypatch: MonkeyPatch, tmp_path: Path,
) -> None:
    """Ensure the proofread stage runs and analysis gets the corrected file."""
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
        Path(cmd_list[-1]).write_text("mp3")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pipeline.subprocess, "run", fake_ffmpeg_run)
    monkeypatch.setattr(pipeline, "ffmpeg_bin", lambda: "ffmpeg")

    def fake_transcribe_file(config: pipeline.TranscribeConfig) -> Path:
        out_path = config.outdir / config.input_path.with_suffix(".json").name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({
                "language": "ru",
                "duration": 10.0,
                "segments": [{"start": 0.0, "end": 10.0, "text": "привет мир"}],
            }),
            encoding="utf-8",
        )
        out_path.with_suffix(".srt").write_text(
            "1\n00:00:00,000 --> 00:00:10,000\nпривет мир\n",
            encoding="utf-8",
        )
        return out_path

    monkeypatch.setattr(pipeline, "transcribe_file", fake_transcribe_file)

    proofread_calls: list[dict[str, object]] = []

    async def fake_run_proofread(
        *,
        transcript_path: Path,
        output_path: Path,
        url: str,
        model: str,
        proofread_conf: dict[str, object],
        prompts_conf: dict[str, object],
        quiet: bool = False,
        verbose: bool = False,
    ) -> Path:
        proofread_calls.append({
            "transcript_path": transcript_path,
            "output_path": output_path,
            "model": model,
            "proofread_conf": proofread_conf,
        })
        output_path.write_text(
            json.dumps({
                "language": "ru",
                "segments": [{"start": 0.0, "end": 10.0, "text": "Привет, мир!"}],
            }),
            encoding="utf-8",
        )
        output_path.with_suffix(".srt").write_text(
            "1\n00:00:00,000 --> 00:00:10,000\nПривет, мир!\n",
            encoding="utf-8",
        )
        return output_path

    monkeypatch.setattr(pipeline, "run_proofread", fake_run_proofread)

    analysis_calls: list[dict[str, object]] = []

    async def fake_run_staged_analysis(**kwargs: object) -> list[dict[str, object]]:
        analysis_calls.append(kwargs)
        outdir = kwargs["outdir"]
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "moments.json").write_text("[]", encoding="utf-8")
        (outdir / "reels.md").write_text("# Reels Suggestions\n", encoding="utf-8")
        return []

    monkeypatch.setattr(pipeline, "run_staged_analysis", fake_run_staged_analysis)
    monkeypatch.setattr(pipeline, "run_module", lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, "llama_cpp_start", lambda **kw: None)
    monkeypatch.setattr(pipeline, "llama_cpp_stop", lambda proc: None)
    monkeypatch.setattr(pipeline, "wait_for_server_ready", lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, "_kill_llama_server", lambda *a: None)
    monkeypatch.setattr(pipeline, "is_tcp_open", lambda *a: False)

    conf = {
        "paths": {"input_dir": str(input_dir), "output_dir": str(tmp_path / "output")},
        "transcription": {"language": "ru", "device": "cpu"},
        "llama_cpp": {
            "roles": {
                "scout": "gemma4",
                "cleanup_refine": "gemma4",
                "judge_metadata": "gemma4",
                "proofread": "gemma4:26b",
            },
            "url": "http://127.0.0.1:8080/v1/chat/completions",
            "service": {"auto_start": True, "model_path": str(tmp_path / "model.gguf")},
        },
        "processing": {"reels_count": 1, "reel_padding": 5},
        "video": {"threads": 1},
        "exports": {},
        "subtitles": {"enabled": False},
        "diarization": {"enabled": False},
        "proofread": {"enabled": True, "max_chars_chunk": 4000},
        "prompts": {"language": "auto", "variant": "default"},
    }

    pipeline.run_pipeline(conf=conf, repo_dir=tmp_path, quiet=True, verbose=False)

    assert len(proofread_calls) == 1
    expected_proofread = tmp_path / "output" / "video" / "video.proofread.json"
    assert proofread_calls[0]["transcript_path"] == tmp_path / "output" / "video" / "video.json"
    assert proofread_calls[0]["output_path"] == expected_proofread
    assert proofread_calls[0]["model"] == "gemma4:26b"

    assert len(analysis_calls) == 1
    assert analysis_calls[0]["transcript_path"] == expected_proofread


def test_run_pipeline_burns_proofread_transcript_into_reels(
    monkeypatch: MonkeyPatch, tmp_path: Path,
) -> None:
    """The burn-in stages must receive the proofread transcript, not the raw one."""
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
        _write_ffmpeg_outputs(list(cmd))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pipeline.subprocess, "run", fake_ffmpeg_run)
    monkeypatch.setattr(pipeline, "ffmpeg_bin", lambda: "ffmpeg")

    def fake_transcribe_file(config: pipeline.TranscribeConfig) -> Path:
        out_path = config.outdir / config.input_path.with_suffix(".json").name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({
                "language": "ru",
                "segments": [{"start": 0.0, "end": 10.0, "text": "привет мир"}],
            }),
            encoding="utf-8",
        )
        out_path.with_suffix(".srt").write_text("1\n", encoding="utf-8")
        return out_path

    monkeypatch.setattr(pipeline, "transcribe_file", fake_transcribe_file)

    async def fake_run_proofread(*, transcript_path: Path, output_path: Path, **_: object) -> Path:
        output_path.write_text(
            json.dumps({
                "language": "ru",
                "segments": [{"start": 0.0, "end": 10.0, "text": "Привет, мир!"}],
            }),
            encoding="utf-8",
        )
        output_path.with_suffix(".srt").write_text("1\n", encoding="utf-8")
        return output_path

    monkeypatch.setattr(pipeline, "run_proofread", fake_run_proofread)

    async def fake_run_staged_analysis(**kwargs: object) -> list[dict[str, object]]:
        outdir = kwargs["outdir"]
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "moments.json").write_text(
            json.dumps([{"start": 1.0, "end": 5.0, "title": "T", "score": 9}]),
            encoding="utf-8",
        )
        (outdir / "reels.md").write_text("# Reels Suggestions\n", encoding="utf-8")
        return []

    monkeypatch.setattr(pipeline, "run_staged_analysis", fake_run_staged_analysis)

    module_calls: list[tuple[str, list[str]]] = []

    def fake_run_module(module: str, args: list[str], **_: object) -> None:
        module_calls.append((module, list(args)))

    monkeypatch.setattr(pipeline, "run_module", fake_run_module)
    monkeypatch.setattr(pipeline, "sync_reel_markdowns", lambda *a, **kw: [])
    monkeypatch.setattr(pipeline, "llama_cpp_start", lambda **kw: None)
    monkeypatch.setattr(pipeline, "llama_cpp_stop", lambda proc: None)
    monkeypatch.setattr(pipeline, "wait_for_server_ready", lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, "_kill_llama_server", lambda *a: None)
    monkeypatch.setattr(pipeline, "is_tcp_open", lambda *a: False)

    conf = {
        "paths": {"input_dir": str(input_dir), "output_dir": str(tmp_path / "output")},
        "transcription": {"language": "ru", "device": "cpu"},
        "llama_cpp": {
            "roles": {
                "scout": "gemma4",
                "cleanup_refine": "gemma4",
                "judge_metadata": "gemma4",
                "proofread": "gemma4",
            },
            "url": "http://127.0.0.1:8080/v1/chat/completions",
            "service": {"auto_start": True, "model_path": str(tmp_path / "model.gguf")},
        },
        "processing": {"reels_count": 1, "reel_padding": 5},
        "video": {"threads": 1},
        "exports": {},
        "subtitles": {"enabled": True},
        "diarization": {"enabled": False},
        "proofread": {"enabled": True},
        "prompts": {"language": "auto", "variant": "default"},
    }

    pipeline.run_pipeline(conf=conf, repo_dir=tmp_path, quiet=True, verbose=False)

    expected_proofread = tmp_path / "output" / "video" / "video.proofread.json"
    assert expected_proofread.exists()

    cut_args = [args for module, args in module_calls if module.endswith("video_processor")]
    assert len(cut_args) == 1
    args = cut_args[0]
    assert "--burn-subtitles" in args
    # The subtitle source handed to the burn stage is the corrected transcript.
    assert args[args.index("--transcript-json") + 1] == str(expected_proofread)


def test_run_pipeline_resyncs_cached_reels_from_proofread_transcript(
    monkeypatch: MonkeyPatch, tmp_path: Path,
) -> None:
    """Cached reels get their subtitle sidecars rebuilt from the proofread file."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "video.mp4").write_text("x")
    (input_dir / "video.mp3").write_text("mp3")
    (input_dir / "video.wav").write_text("wav")

    monkeypatch.setattr(pipeline, "ffmpeg_bin", lambda: "ffmpeg")

    output_root = tmp_path / "output"
    episode_dir = output_root / "video"
    model_dir = episode_dir / "gemma4"
    reels_dir = model_dir / "reels"
    reels_dir.mkdir(parents=True)

    # Everything is already cached, including the proofread transcript.
    (episode_dir / "video.json").write_text(
        json.dumps({"language": "ru", "segments": []}), encoding="utf-8",
    )
    (episode_dir / "video.srt").write_text("1\n", encoding="utf-8")
    proofread_path = episode_dir / "video.proofread.json"
    proofread_path.write_text(
        json.dumps({"language": "ru", "segments": []}), encoding="utf-8",
    )
    (episode_dir / "video.proofread.srt").write_text("1\n", encoding="utf-8")
    (model_dir / "moments.json").write_text(
        json.dumps([{"start": 1.0, "end": 5.0, "title": "T", "score": 9}]),
        encoding="utf-8",
    )
    (model_dir / "reels.md").write_text("# Reels Suggestions\n", encoding="utf-8")
    (reels_dir / "reel_01.mp4").write_text("mp4")

    sync_calls: list[Path] = []

    def fake_sync_reel_burned_subtitles(
        moments: list[dict[str, object]],
        reels_root: Path,
        *,
        transcript_json_path: Path,
        padding: float,
        settings: object,
        verbose: bool = False,
    ) -> list[Path]:
        sync_calls.append(transcript_json_path)
        return []

    monkeypatch.setattr(pipeline, "sync_reel_burned_subtitles", fake_sync_reel_burned_subtitles)
    monkeypatch.setattr(pipeline, "sync_reel_markdowns", lambda *a, **kw: [])
    monkeypatch.setattr(pipeline, "run_module", lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, "llama_cpp_start", lambda **kw: None)
    monkeypatch.setattr(pipeline, "llama_cpp_stop", lambda proc: None)
    monkeypatch.setattr(pipeline, "wait_for_server_ready", lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, "_kill_llama_server", lambda *a: None)
    monkeypatch.setattr(pipeline, "is_tcp_open", lambda *a: False)

    conf = {
        "paths": {"input_dir": str(input_dir), "output_dir": str(output_root)},
        "transcription": {"language": "ru", "device": "cpu"},
        "llama_cpp": {
            "roles": {
                "scout": "gemma4",
                "cleanup_refine": "gemma4",
                "judge_metadata": "gemma4",
                "proofread": "gemma4",
            },
            "url": "http://127.0.0.1:8080/v1/chat/completions",
            "service": {"auto_start": True, "model_path": str(tmp_path / "model.gguf")},
        },
        "processing": {"reels_count": 1, "reel_padding": 5},
        "video": {"threads": 1},
        "exports": {},
        "subtitles": {"enabled": True},
        "diarization": {"enabled": False},
        "proofread": {"enabled": True},
        "prompts": {"language": "auto", "variant": "default"},
    }

    pipeline.run_pipeline(conf=conf, repo_dir=tmp_path, quiet=True, verbose=False)

    assert sync_calls == [proofread_path]


def test_run_pipeline_builds_the_article_from_the_proofread_transcript(
    monkeypatch: MonkeyPatch, tmp_path: Path,
) -> None:
    """The article stage runs after proofreading and reads the corrected text."""
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
        _write_ffmpeg_outputs(list(cmd))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pipeline.subprocess, "run", fake_ffmpeg_run)
    monkeypatch.setattr(pipeline, "ffmpeg_bin", lambda: "ffmpeg")

    def fake_transcribe_file(config: pipeline.TranscribeConfig) -> Path:
        out_path = config.outdir / config.input_path.with_suffix(".json").name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"language": "ru", "segments": [
                {"start": 0.0, "end": 10.0, "text": "привет мир"},
            ]}),
            encoding="utf-8",
        )
        out_path.with_suffix(".srt").write_text("1\n", encoding="utf-8")
        return out_path

    monkeypatch.setattr(pipeline, "transcribe_file", fake_transcribe_file)

    async def fake_run_proofread(*, transcript_path: Path, output_path: Path, **_: object) -> Path:
        output_path.write_text(
            json.dumps({"language": "ru", "segments": [
                {"start": 0.0, "end": 10.0, "text": "Привет, мир!"},
            ]}),
            encoding="utf-8",
        )
        output_path.with_suffix(".srt").write_text("1\n", encoding="utf-8")
        return output_path

    monkeypatch.setattr(pipeline, "run_proofread", fake_run_proofread)

    article_calls: list[dict[str, object]] = []

    async def fake_run_article(
        *,
        transcript_path: Path,
        output_path: Path,
        url: str,
        model: str,
        article_conf: dict[str, object],
        prompts_conf: dict[str, object],
        diarization_path: Path | None = None,
        title: str | None = None,
        quiet: bool = False,
        verbose: bool = False,
    ) -> Path:
        article_calls.append({
            "transcript_path": transcript_path,
            "output_path": output_path,
            "model": model,
            "title": title,
        })
        output_path.write_text("# video\n\n## Раздел\n\nАбзац.\n", encoding="utf-8")
        output_path.with_suffix(".json").write_text("{}", encoding="utf-8")
        return output_path

    monkeypatch.setattr(pipeline, "run_article", fake_run_article)

    async def fake_run_staged_analysis(**kwargs: object) -> list[dict[str, object]]:
        outdir = kwargs["outdir"]
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "moments.json").write_text("[]", encoding="utf-8")
        (outdir / "reels.md").write_text("# Reels Suggestions\n", encoding="utf-8")
        return []

    monkeypatch.setattr(pipeline, "run_staged_analysis", fake_run_staged_analysis)
    monkeypatch.setattr(pipeline, "run_module", lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, "sync_reel_markdowns", lambda *a, **kw: [])
    monkeypatch.setattr(pipeline, "llama_cpp_start", lambda **kw: None)
    monkeypatch.setattr(pipeline, "llama_cpp_stop", lambda proc: None)
    monkeypatch.setattr(pipeline, "wait_for_server_ready", lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, "_kill_llama_server", lambda *a: None)
    monkeypatch.setattr(pipeline, "is_tcp_open", lambda *a: False)

    conf = {
        "paths": {"input_dir": str(input_dir), "output_dir": str(tmp_path / "output")},
        "transcription": {"language": "ru", "device": "cpu"},
        "llama_cpp": {
            "roles": {
                "scout": "gemma4",
                "cleanup_refine": "gemma4",
                "judge_metadata": "gemma4",
                "proofread": "gemma4",
                "article": "gemma4:12b",
            },
            "url": "http://127.0.0.1:8080/v1/chat/completions",
            "service": {"auto_start": True, "model_path": str(tmp_path / "model.gguf")},
        },
        "processing": {"reels_count": 1, "reel_padding": 5},
        "video": {"threads": 1},
        "exports": {},
        "subtitles": {"enabled": False},
        "diarization": {"enabled": False},
        "proofread": {"enabled": True},
        "article": {"enabled": True},
        "prompts": {"language": "auto", "variant": "default"},
    }

    pipeline.run_pipeline(conf=conf, repo_dir=tmp_path, quiet=True, verbose=False)

    assert len(article_calls) == 1
    episode_dir = tmp_path / "output" / "video"
    # It retells the corrected transcript, not the raw one.
    assert article_calls[0]["transcript_path"] == episode_dir / "video.proofread.json"
    # ".proofread" must not leak into the article file name.
    assert article_calls[0]["output_path"] == episode_dir / "video.article.md"
    assert article_calls[0]["model"] == "gemma4:12b"
    assert article_calls[0]["title"] == "video"
    assert (episode_dir / "video.article.md").exists()


def test_resolve_stages_only_skip_and_typos() -> None:
    """--only/--skip must be explicit, and a typo must fail loudly."""
    import pytest

    assert pipeline.resolve_stages() == set(pipeline.PIPELINE_STAGES)
    assert pipeline.resolve_stages(only="proofread,article") == {"proofread", "article"}
    assert pipeline.resolve_stages(only=["article"]) == {"article"}

    without_cut = pipeline.resolve_stages(skip="cut")
    assert "cut" not in without_cut
    assert "analyze" in without_cut

    # Whitespace and case are tolerated; --skip wins over --only.
    assert pipeline.resolve_stages(only=" Proofread , Article ", skip="article") == {"proofread"}

    # A typo must not silently drop half the pipeline.
    with pytest.raises(SystemExit, match="proofred"):
        pipeline.resolve_stages(only="proofred")
    with pytest.raises(SystemExit, match="No pipeline stages left"):
        pipeline.resolve_stages(only="cut", skip="cut")


def test_run_pipeline_only_article_skips_the_other_stages(
    monkeypatch: MonkeyPatch, tmp_path: Path,
) -> None:
    """--only article must run just that stage over an existing transcript."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "video.mp4").write_text("x")
    (input_dir / "video.mp3").write_text("mp3")
    (input_dir / "video.wav").write_text("wav")

    monkeypatch.setattr(pipeline, "ffmpeg_bin", lambda: "ffmpeg")

    output_root = tmp_path / "output"
    episode_dir = output_root / "video"
    episode_dir.mkdir(parents=True)
    (episode_dir / "video.json").write_text(
        json.dumps({"language": "ru", "segments": [
            {"start": 0.0, "end": 10.0, "text": "сырой текст"},
        ]}),
        encoding="utf-8",
    )
    (episode_dir / "video.srt").write_text("1\n", encoding="utf-8")
    # A proofread transcript exists from an earlier run.
    (episode_dir / "video.proofread.json").write_text(
        json.dumps({"language": "ru", "segments": [
            {"start": 0.0, "end": 10.0, "text": "Исправленный текст."},
        ]}),
        encoding="utf-8",
    )
    (episode_dir / "video.proofread.srt").write_text("1\n", encoding="utf-8")

    called: dict[str, object] = {}

    def fail(name: str):
        def _fail(*a: object, **kw: object) -> None:
            called[name] = True
            raise AssertionError(f"{name} must not run under --only article")
        return _fail

    monkeypatch.setattr(pipeline, "transcribe_file", fail("transcribe"))
    monkeypatch.setattr(pipeline, "run_proofread", fail("proofread"))
    monkeypatch.setattr(pipeline, "run_staged_analysis", fail("analyze"))
    monkeypatch.setattr(pipeline, "run_module", fail("cut"))

    article_calls: list[Path] = []

    async def fake_run_article(*, transcript_path: Path, output_path: Path, **_: object) -> Path:
        article_calls.append(transcript_path)
        output_path.write_text("# video\n\n## Раздел\n\nАбзац.\n", encoding="utf-8")
        output_path.with_suffix(".json").write_text("{}", encoding="utf-8")
        return output_path

    monkeypatch.setattr(pipeline, "run_article", fake_run_article)
    monkeypatch.setattr(pipeline, "sync_reel_markdowns", lambda *a, **kw: [])
    monkeypatch.setattr(pipeline, "llama_cpp_start", lambda **kw: None)
    monkeypatch.setattr(pipeline, "llama_cpp_stop", lambda proc: None)
    monkeypatch.setattr(pipeline, "wait_for_server_ready", lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, "_kill_llama_server", lambda *a: None)
    monkeypatch.setattr(pipeline, "is_tcp_open", lambda *a: False)

    conf = {
        "paths": {"input_dir": str(input_dir), "output_dir": str(output_root)},
        "transcription": {"language": "ru", "device": "cpu"},
        "llama_cpp": {
            "roles": {
                "scout": "gemma4", "cleanup_refine": "gemma4",
                "judge_metadata": "gemma4", "proofread": "gemma4", "article": "gemma4",
            },
            "url": "http://127.0.0.1:8080/v1/chat/completions",
            "service": {"auto_start": True, "model_path": str(tmp_path / "model.gguf")},
        },
        "processing": {"reels_count": 1, "reel_padding": 5},
        "video": {"threads": 1},
        "exports": {},
        "subtitles": {"enabled": False},
        "diarization": {"enabled": True},
        "proofread": {"enabled": True},
        "article": {"enabled": True},
        "prompts": {"language": "auto", "variant": "default"},
    }

    pipeline.run_pipeline(
        conf=conf, repo_dir=tmp_path, quiet=True, verbose=False,
        stages={"article"},
    )

    assert called == {}, "no other stage may run"
    assert len(article_calls) == 1
    # Even though proofreading did not run now, its output is what gets edited.
    assert article_calls[0] == episode_dir / "video.proofread.json"
