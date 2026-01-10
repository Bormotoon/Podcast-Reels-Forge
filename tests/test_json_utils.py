"""Tests for JSON extraction helper."""

import pytest

from podcast_reels_forge.utils.json_utils import extract_first_json_object


def test_extract_first_json_object_simple() -> None:
    """Return the first JSON object embedded in text."""
    obj = extract_first_json_object('hello {"a": 1, "b": {"c": 2}} world')
    if obj != {"a": 1, "b": {"c": 2}}:
        message = "Parsed JSON does not match expected structure"
        raise AssertionError(message)


def test_extract_first_json_object_raises_when_missing() -> None:
    """Raise ValueError when no JSON object is found."""
    with pytest.raises(ValueError, match="Could not find JSON object"):
        extract_first_json_object("no json here")
