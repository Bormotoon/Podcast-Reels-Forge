"""RU: Утилиты для извлечения JSON из неструктурированного текста.

EN: Utilities for extracting JSON from unstructured text.
"""

from __future__ import annotations

import json
import re
from json import JSONDecodeError
from typing import Any


def extract_first_json_value(text: str) -> dict[str, Any] | list[Any]:
    """RU: Ищет и парсит первый корректный JSON-объект *или* массив в тексте.

    EN: Find and parse the first valid JSON object *or* array in the text.
    """
    # Try to find JSON in markdown code blocks first
    code_block_pattern = r"```(?:json)?\s*([\{\[][\s\S]*?[\}\]])\s*```"
    match = re.search(code_block_pattern, text)
    if match:
        try:
            return json.loads(match.group(1))
        except JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch not in ("{", "["):
            continue
        try:
            obj, _ = decoder.raw_decode(text, idx)
            if isinstance(obj, (dict, list)):
                return obj
        except JSONDecodeError:
            continue

    message = "Could not find JSON object or array in text"
    raise ValueError(message)


def extract_first_json_object(text: str) -> dict[str, Any]:
    """RU: Ищет и парсит первый корректный JSON-объект в тексте.

    Важно: намеренно пропускает JSON-массивы ([]) и продолжает поиск следующего объекта.

    EN: Find and parse the first valid JSON object (dict) in the text.

    Note: intentionally skips JSON arrays ([]) and keeps searching for the next object.
    """
    # Try to find JSON in markdown code blocks first
    code_block_pattern = r"```(?:json)?\s*(\{[\s\S]*?\})\s*```"
    match = re.search(code_block_pattern, text)
    if match:
        try:
            obj = json.loads(match.group(1))
            if isinstance(obj, dict):
                return obj
        except JSONDecodeError:
            pass

    # Fall back to scanning for '{' only
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, idx)
            if isinstance(obj, dict):
                return obj
        except JSONDecodeError:
            continue

    message = "Could not find JSON object in text"
    raise ValueError(message)
