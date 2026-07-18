"""RU: Утилиты для извлечения JSON из неструктурированного текста.

EN: Utilities for extracting JSON from unstructured text.
"""

from __future__ import annotations

import json
from typing import Any

_CLOSERS = {"{": "}", "[": "]"}


def _strip_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _repair_truncated_json(text: str) -> dict[str, Any] | list[Any] | None:
    """RU: Пытается спасти усечённый (оборванный) JSON.

    EN: Attempt to salvage a truncated JSON value by dropping the trailing
    incomplete element and closing any still-open objects/arrays.

    Local LLMs (llama.cpp) that hit their ``n_predict`` token budget stop
    mid-token, leaving the JSON unterminated. Instead of losing the whole
    response we keep every element that was fully emitted before the cut.
    """
    stack: list[str] = []
    in_string = False
    escape = False
    # Candidate cut points: (index, snapshot of the open-bracket stack). A cut
    # is safe right after a completed container ("}"/"]") or right before a
    # comma that separates elements (dropping whatever partial element follows).
    boundaries: list[tuple[int, tuple[str, ...]]] = []

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            boundaries.append((i + 1, tuple(stack)))
        elif ch == "," and stack:
            boundaries.append((i, tuple(stack)))

    # Try the furthest safe boundary first so we recover as much as possible.
    for cut, snap in reversed(boundaries):
        candidate = text[:cut].rstrip()
        candidate += "".join(_CLOSERS[c] for c in reversed(snap))
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, (dict, list)):
            return value
    return None


def extract_first_json_value(text: str) -> dict[str, Any] | list[Any]:
    """RU: Извлекает первый корректный JSON-объект или массив из текста.

    EN: Extract the first valid JSON object or array from the text.

    Tolerates prose before/after the JSON and salvages truncated output that
    was cut off by the model's token limit.
    """
    cleaned = _strip_fences(text)

    start = -1
    for idx, ch in enumerate(cleaned):
        if ch in "{[":
            start = idx
            break
    if start < 0:
        raise ValueError("Could not parse JSON: no object or array found")

    body = cleaned[start:]

    # Fast path + tolerate trailing prose: raw_decode reads the first complete
    # JSON value and ignores anything after it.
    try:
        value, _ = json.JSONDecoder().raw_decode(body)
    except json.JSONDecodeError as first_error:
        repaired = _repair_truncated_json(body)
        if repaired is not None:
            return repaired
        raise ValueError(f"Could not parse JSON: {first_error}")

    if isinstance(value, (dict, list)):
        return value
    raise ValueError("Could not parse JSON: top-level value is not object or array")
