"""RU: Утилиты для извлечения JSON из неструктурированного текста.

EN: Utilities for extracting JSON from unstructured text.
"""

from __future__ import annotations

import json
import re
from json import JSONDecodeError
from typing import Any


def extract_first_json_object(text: str) -> dict[str, Any]:
    """RU: Ищет и парсит первый корректный JSON-объект в тексте.

    EN: Find and parse the first valid JSON object in the text.
    """
    # Try to find JSON in markdown code blocks first
    code_block_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
    match = re.search(code_block_pattern, text)
    if match:
        try:
            obj = json.loads(match.group(1))
            if isinstance(obj, dict):
                return obj
        except JSONDecodeError:
            pass

    # Fall back to scanning for { character
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
