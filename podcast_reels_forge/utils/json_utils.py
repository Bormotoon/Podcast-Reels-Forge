"""RU: Утилиты для извлечения JSON из неструктурированного текста.

EN: Utilities for extracting JSON from unstructured text.
"""

from __future__ import annotations

import json
from typing import Any


def extract_first_json_value(text: str) -> dict[str, Any] | list[Any]:
    """RU: Извлекает первый корректный JSON-объект или массив из текста.

    EN: Extract the first valid JSON object or array from the text.
    """
    cleaned = text.strip()
    # Remove markdown formatting if the model slipped it in
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse JSON: {e}")
