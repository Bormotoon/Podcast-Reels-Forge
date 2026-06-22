"""Tests for JSON extraction helper."""

import pytest

from podcast_reels_forge.utils.json_utils import extract_first_json_value


def test_extract_first_json_value_simple() -> None:
    """Return the first JSON value embedded in text."""
    obj = extract_first_json_value('hello {"a": 1, "b": {"c": 2}} world')
    assert obj == {"a": 1, "b": {"c": 2}}


def test_extract_first_json_value_array() -> None:
    """Return a JSON array when it appears first."""
    obj = extract_first_json_value('[1, 2, 3] and {"a": 1}')
    assert obj == [1, 2, 3]


def test_extract_first_json_value_with_newlines() -> None:
    """Handle multiline JSON."""
    text = """
    Result:
    {
      "key": "value",
      "list": [1, 2]
    }
    """
    obj = extract_first_json_value(text)
    assert obj == {"key": "value", "list": [1, 2]}


def test_extract_first_json_value_markdown_block() -> None:
    """Handle markdown code blocks."""
    text = '```json\n{"a": 1}\n```'
    obj = extract_first_json_value(text)
    assert obj == {"a": 1}


def test_extract_first_json_value_raises_when_missing() -> None:
    """Raise ValueError when no JSON is found."""
    with pytest.raises(ValueError, match="Could not parse JSON"):
        extract_first_json_value("no json here")
