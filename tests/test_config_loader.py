"""Tests for config.yaml overlays (extends + config.local.yaml)."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_reels_forge.utils.config_loader import deep_merge, load_config, load_config_with_sources


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_deep_merge_merges_mappings_and_replaces_the_rest() -> None:
    base = {"a": {"x": 1, "y": 2}, "list": [1, 2], "keep": True}
    merged = deep_merge(base, {"a": {"y": 3}, "list": [9]})
    assert merged == {"a": {"x": 1, "y": 3}, "list": [9], "keep": True}
    assert base["a"] == {"x": 1, "y": 2}, "the base is not mutated"


def test_plain_config_loads_as_is(tmp_path: Path) -> None:
    cfg = _write(tmp_path / "config.yaml", "paths:\n  input_dir: input\n")
    conf, sources = load_config_with_sources(cfg)
    assert conf == {"paths": {"input_dir": "input"}}
    assert sources == [cfg]


def test_local_overlay_is_merged_last(tmp_path: Path) -> None:
    cfg = _write(tmp_path / "config.yaml", "llama_cpp:\n  cache_ram_mb: auto\n  ctx: 8192\n")
    _write(tmp_path / "config.local.yaml", "llama_cpp:\n  cache_ram_mb: 1024\n")
    conf = load_config(cfg)
    assert conf["llama_cpp"] == {"cache_ram_mb": 1024, "ctx": 8192}


def test_extends_builds_on_the_base_and_keeps_the_local_overlay(tmp_path: Path) -> None:
    _write(tmp_path / "config.yaml", "paths:\n  input_dir: input\n  output_dir: output\nx: 1\n")
    _write(tmp_path / "config.local.yaml", "x: 2\n")
    pos = _write(tmp_path / "config.pos.yaml", "extends: config.yaml\npaths:\n  input_dir: input_pos\n")
    conf, sources = load_config_with_sources(pos)
    assert conf == {"paths": {"input_dir": "input_pos", "output_dir": "output"}, "x": 2}
    assert [s.name for s in sources] == ["config.yaml", "config.pos.yaml", "config.local.yaml"]


def test_extends_cycle_is_an_error(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.yaml", "extends: b.yaml\n")
    _write(tmp_path / "b.yaml", "extends: a.yaml\n")
    with pytest.raises(ValueError, match="too deep"):
        load_config(a)


def test_local_overlay_found_next_to_the_extends_root(tmp_path: Path) -> None:
    _write(tmp_path / "config.yaml", "x: 1\ny: 1\n")
    _write(tmp_path / "config.local.yaml", "x: 2\n")
    (tmp_path / "local").mkdir()
    night = _write(tmp_path / "local" / "night.yaml", "extends: ../config.yaml\ny: 3\n")
    conf, sources = load_config_with_sources(night)
    assert conf == {"x": 2, "y": 3}
    assert sources[-1].name == "config.local.yaml"


def test_hf_token_accepts_the_standard_names(monkeypatch: pytest.MonkeyPatch) -> None:
    from podcast_reels_forge.utils.env import HF_TOKEN_VARS, hf_token

    for name in HF_TOKEN_VARS:
        monkeypatch.delenv(name, raising=False)
    assert hf_token() is None
    monkeypatch.setenv("HUGGING_FACE_ACCESS_TOKEN", "hf_abc")
    assert hf_token() == "hf_abc"
    monkeypatch.setenv("PYANNOTE_TOKEN", "own")
    assert hf_token() == "own"
