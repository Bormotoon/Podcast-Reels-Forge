"""RU: JSON-схемы ответов LLM для грамматики llama.cpp.

llama.cpp конвертирует ``json_schema`` в GBNF-грамматику и сэмплирует строго
по ней, поэтому схема должна оставаться в поддерживаемом подмножестве:
простые типы, ``properties``, ``required``, ``items``. Без ``$ref``, ``oneOf``
и прочего — иначе сервер отвечает 400.

EN: JSON schemas for LLM responses, used as llama.cpp grammars.

llama.cpp converts ``json_schema`` into a GBNF grammar and samples strictly
against it, so the schema has to stay inside the supported subset: plain
types, ``properties``, ``required`` and ``items``. No ``$ref``, ``oneOf`` or
similar — the server answers 400 on those.
"""

from __future__ import annotations

from typing import Any

# RU: Схема одного момента. `required` держим минимальным: чем больше
# обязательных полей, тем длиннее ответ и тем выше шанс упереться в n_predict.
# EN: Schema of a single moment. `required` is kept minimal: every mandatory
# field lengthens the answer and brings the n_predict budget closer.
MOMENT_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "start": {"type": "number"},
        "end": {"type": "number"},
        "clip_type": {"type": "string"},
        "title": {"type": "string"},
        "quote": {"type": "string"},
        "why": {"type": "string"},
        "score": {"type": "number"},
        "hook": {"type": "string"},
        "caption": {"type": "string"},
        "hashtags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["start", "end", "title", "score"],
}

MOMENTS_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "moments": {"type": "array", "items": MOMENT_ITEM_SCHEMA},
    },
    "required": ["moments"],
}

# RU: Схема-заглушка «любой объект» — исходное поведение и путь отката для
# сборок llama.cpp, которые не переваривают полную схему.
# EN: The permissive "any object" schema — the original behaviour and the
# downgrade path for llama.cpp builds that reject the full one.
ANY_OBJECT_SCHEMA: dict[str, Any] = {"type": "object"}
