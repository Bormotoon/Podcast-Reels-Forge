"""Tests for JSON extraction helper."""

import pytest

from podcast_reels_forge.utils.json_utils import extract_first_json_object


def test_extract_first_json_object_simple() -> None:
    """Return the first JSON object embedded in text."""
    obj = extract_first_json_object('hello {"a": 1, "b": {"c": 2}} world')
    assert obj == {"a": 1, "b": {"c": 2}}


def test_extract_first_json_object_multiple() -> None:
    """Return only the first JSON object."""
    obj = extract_first_json_object('first {"a": 1} then {"b": 2}')
    assert obj == {"a": 1}


def test_extract_first_json_object_with_newlines() -> None:
    """Handle multiline JSON."""
    text = """
    Result:
    {
      "key": "value",
      "list": [1, 2]
    }
    """
    obj = extract_first_json_object(text)
    assert obj == {"key": "value", "list": [1, 2]}


def test_extract_first_json_object_skips_non_dict_prefix() -> None:
    """Ensure it skips lists and continues searching for a dictionary."""
    text = 'here is a list [1, 2, 3] and then a dict {"a": 1}'
    obj = extract_first_json_object(text)
    assert obj == {"a": 1}


def test_extract_first_json_object_raises_when_missing() -> None:
    """Raise ValueError when no JSON object is found."""
    with pytest.raises(ValueError, match="Could not find JSON object"):
        extract_first_json_object("no json here")

