"""Tests for the minimal `.env` loader."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from podcast_reels_forge.utils.env import load_dotenv, parse_dotenv

if TYPE_CHECKING:
    import pytest

    MonkeyPatch = pytest.MonkeyPatch


def test_parse_handles_the_shapes_a_real_env_file_has() -> None:
    """Quotes, `export`, comments and blank lines all appear in the wild."""
    parsed = parse_dotenv(
        "\n".join([
            "# a comment",
            "",
            'YOUTUBE_API_KEY="AIzaSyExample"',
            "export PYANNOTE_TOKEN=hf_plain",
            "SINGLE='quoted'",
            "  SPACED = value with spaces  ",
            "EMPTY=",
            "not a pair",
            "URL=https://example.com/path?a=1&b=2",
        ]),
    )

    assert parsed["YOUTUBE_API_KEY"] == "AIzaSyExample"
    assert parsed["PYANNOTE_TOKEN"] == "hf_plain"
    assert parsed["SINGLE"] == "quoted"
    assert parsed["SPACED"] == "value with spaces"
    assert parsed["EMPTY"] == ""
    # A value containing '=' must survive intact, not be cut at the first one.
    assert parsed["URL"] == "https://example.com/path?a=1&b=2"
    assert "not a pair" not in parsed


def test_a_real_environment_variable_beats_the_file(
    monkeypatch: MonkeyPatch, tmp_path: Path,
) -> None:
    """`.env` supplies defaults; it must not override a deliberate export."""
    (tmp_path / ".env").write_text("YOUTUBE_API_KEY=from_file\n", encoding="utf-8")
    monkeypatch.setenv("YOUTUBE_API_KEY", "from_shell")

    applied = load_dotenv(tmp_path)

    import os

    assert os.environ["YOUTUBE_API_KEY"] == "from_shell"
    assert applied == {}


def test_missing_env_file_is_not_an_error(tmp_path: Path) -> None:
    """Most installs have no `.env` at all; that is the normal case."""
    assert load_dotenv(tmp_path) == {}


def test_values_reach_the_environment(
    monkeypatch: MonkeyPatch, tmp_path: Path,
) -> None:
    (tmp_path / ".env").write_text('YOUTUBE_API_KEY="key123"\n', encoding="utf-8")
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)

    applied = load_dotenv(tmp_path)

    import os

    assert os.environ["YOUTUBE_API_KEY"] == "key123"
    assert applied == {"YOUTUBE_API_KEY": "key123"}
