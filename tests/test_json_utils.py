"""Tests for JSON extraction helper."""

import pytest

from podcast_reels_forge.utils.json_utils import extract_first_json_value


def test_extract_first_json_value_simple() -> None:
    """Return a JSON object from text."""
    obj = extract_first_json_value('{"a": 1, "b": {"c": 2}}')
    assert obj == {"a": 1, "b": {"c": 2}}


def test_extract_first_json_value_array() -> None:
    """Return a JSON array."""
    obj = extract_first_json_value('[1, 2, 3]')
    assert obj == [1, 2, 3]


def test_extract_first_json_value_with_newlines() -> None:
    """Handle multiline JSON."""
    text = """
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


def test_extract_first_json_value_ignores_trailing_prose() -> None:
    """Tolerate explanatory text after the JSON value."""
    obj = extract_first_json_value('{"a": 1}\n\nHope this helps!')
    assert obj == {"a": 1}


def test_extract_first_json_value_ignores_leading_prose() -> None:
    """Tolerate a preamble before the JSON value."""
    obj = extract_first_json_value('Here is the JSON:\n{"a": 1}')
    assert obj == {"a": 1}


def test_extract_first_json_value_salvages_truncated_array() -> None:
    """Recover the elements emitted before the model cut off mid-string.

    The salvage keeps every complete key/value pair, so a partly-built trailing
    object may survive with only its finished fields; downstream validation
    (``coerce_moment_record``) then drops any element missing required fields.
    """
    text = (
        '{\n  "moments": [\n'
        '    {"start": 1.0, "title": "one"},\n'
        '    {"start": 2.0, "title": "two"},\n'
        '    {"start": 3.0, "title": "К чему приведет'
    )
    obj = extract_first_json_value(text)
    assert isinstance(obj, dict)
    moments = obj["moments"]
    assert moments[:2] == [
        {"start": 1.0, "title": "one"},
        {"start": 2.0, "title": "two"},
    ]
    # The truncated third element loses its unterminated "title" string but its
    # completed "start" pair may be recovered.
    assert all("title" not in m or m["title"] in {"one", "two"} for m in moments)


def test_extract_first_json_value_salvages_truncated_after_complete_object() -> None:
    """Recover a fully-closed object even if a trailing comma was emitted."""
    text = '{"moments": [{"a": 1}],'
    obj = extract_first_json_value(text)
    assert obj == {"moments": [{"a": 1}]}
