"""Helpers for per-reel markdown descriptions and hashtags."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

DESCRIPTION_MAX_CHARS = 1000
HASHTAG_COUNT = 5

_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+", re.UNICODE)
_REEL_STEM_RE = re.compile(r"^reel_(\d+)(?:_\d+)?$", re.IGNORECASE)
_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
_HASHTAG_TOKEN_RE = re.compile(r"#?[\wА-Яа-яЁё]+", re.UNICODE)

_SHORT_KEYWORDS = {"ai", "ml", "ux", "vr", "tv", "llm", "gpt"}

_STOPWORDS = {
    # EN
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "for",
    "from",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "him",
    "his",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "like",
    "may",
    "me",
    "more",
    "most",
    "my",
    "no",
    "not",
    "of",
    "on",
    "or",
    "our",
    "out",
    "over",
    "she",
    "so",
    "some",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "under",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    # RU
    "а",
    "без",
    "был",
    "бы",
    "была",
    "были",
    "в",
    "вам",
    "вас",
    "весь",
    "вот",
    "все",
    "всех",
    "всю",
    "где",
    "да",
    "даже",
    "для",
    "до",
    "его",
    "ее",
    "если",
    "еще",
    "же",
    "за",
    "и",
    "из",
    "или",
    "им",
    "их",
    "к",
    "как",
    "какой",
    "когда",
    "кто",
    "ли",
    "лишь",
    "мне",
    "может",
    "мы",
    "на",
    "над",
    "не",
    "него",
    "нее",
    "нет",
    "ни",
    "но",
    "о",
    "об",
    "обо",
    "одна",
    "он",
    "она",
    "они",
    "оно",
    "от",
    "по",
    "под",
    "после",
    "при",
    "про",
    "раз",
    "с",
    "сам",
    "сво",
    "со",
    "так",
    "такие",
    "такой",
    "там",
    "то",
    "тоже",
    "только",
    "тут",
    "у",
    "уже",
    "хотя",
    "чем",
    "через",
    "что",
    "чтобы",
    "эта",
    "это",
    "эти",
    "этот",
    "я",
}


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return _collapse_whitespace(str(value))


def _contains_cyrillic(text: str) -> bool:
    return bool(_CYRILLIC_RE.search(text))


def _trim_text(text: str, max_chars: int) -> str:
    text = _collapse_whitespace(text)
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]

    limit = max_chars - 3
    cut = text[:limit].rstrip()
    if " " in cut:
        space_idx = cut.rfind(" ")
        if space_idx >= int(limit * 0.7):
            cut = cut[:space_idx]
    cut = cut.rstrip(" ,;:.-")
    return f"{cut}..."


def _iter_raw_hashtag_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, Iterable) and not isinstance(
        value,
        (bytes, bytearray),
    ):
        for item in value:
            if item is None:
                continue
            yield str(item)


def normalize_hashtag(raw: Any) -> str | None:
    text = _clean_text(raw)
    if not text:
        return None
    if text.startswith("#"):
        text = text[1:]
    text = re.sub(r"[^\w]+", "", text, flags=re.UNICODE)
    if not text:
        return None
    return f"#{text.lower()}"


def _extract_keywords(text: str) -> list[str]:
    if not text:
        return []

    out: list[str] = []
    for token in _WORD_RE.findall(text.lower()):
        token = token.strip("_")
        if not token or token.isdigit():
            continue
        if len(token) < 2 and token not in _SHORT_KEYWORDS:
            continue
        if token in _STOPWORDS:
            continue
        out.append(token)
    return out


def build_description_text(
    moment: Mapping[str, Any],
    *,
    max_chars: int = DESCRIPTION_MAX_CHARS,
) -> str:
    caption = _clean_text(moment.get("caption"))
    if caption:
        base = caption
    else:
        parts = [
            _clean_text(moment.get(key))
            for key in ("hook", "title", "quote", "why")
        ]
        base = " ".join(part for part in parts if part)

    if not base:
        combined = " ".join(
            _clean_text(moment.get(key))
            for key in ("title", "hook", "quote", "why", "caption")
        ).strip()
        base = "Сильный момент из подкаста." if _contains_cyrillic(combined) else "Strong podcast moment."

    return _trim_text(base, max_chars)


def build_hashtags(
    moment: Mapping[str, Any],
    *,
    count: int = HASHTAG_COUNT,
    description_text: str | None = None,
) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    def add(raw: Any) -> None:
        normalized = normalize_hashtag(raw)
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)

    hashtags_raw = moment.get("hashtags")
    for raw in _iter_raw_hashtag_values(hashtags_raw):
        for token in _HASHTAG_TOKEN_RE.findall(raw):
            add(token)
            if len(out) >= count:
                return out[:count]

    source_texts = [
        description_text or "",
        _clean_text(moment.get("caption")),
        _clean_text(moment.get("hook")),
        _clean_text(moment.get("title")),
        _clean_text(moment.get("quote")),
        _clean_text(moment.get("why")),
    ]
    for source in source_texts:
        for token in _extract_keywords(source):
            add(token)
            if len(out) >= count:
                return out[:count]

    clip_type = _clean_text(moment.get("clip_type")).lower()
    if "story" in clip_type:
        fallback = ["story", "podcast", "reels", "shorts", "viral"]
    elif "highlight" in clip_type or "hot" in clip_type:
        fallback = ["highlight", "podcast", "reels", "shorts", "viral"]
    elif "long" in clip_type:
        fallback = ["longreel", "podcast", "reels", "shorts", "viral"]
    else:
        fallback = ["podcast", "reels", "shorts", "interview", "viral"]

    for token in fallback:
        add(token)
        if len(out) >= count:
            return out[:count]

    return out[:count]


def render_reel_markdown(
    moment: Mapping[str, Any],
    *,
    reel_label: str | None = None,
    max_description_chars: int = DESCRIPTION_MAX_CHARS,
    hashtag_count: int = HASHTAG_COUNT,
) -> str:
    title = _clean_text(moment.get("title")) or _clean_text(reel_label) or "Reel"
    if title.startswith("#"):
        title = title.lstrip("#").strip() or "Reel"

    description = build_description_text(moment, max_chars=max_description_chars)
    hashtags = build_hashtags(
        moment,
        count=hashtag_count,
        description_text=description,
    )

    lines = [
        f"# {title}",
        "",
        description,
        "",
        " ".join(hashtags),
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_reel_markdown(
    moment: Mapping[str, Any],
    reel_path: Path,
    *,
    max_description_chars: int = DESCRIPTION_MAX_CHARS,
    hashtag_count: int = HASHTAG_COUNT,
) -> Path:
    md_path = reel_path.with_suffix(".md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(
        render_reel_markdown(
            moment,
            reel_label=reel_path.stem,
            max_description_chars=max_description_chars,
            hashtag_count=hashtag_count,
        ),
        encoding="utf-8",
    )
    return md_path


def reel_index_from_path(reel_path: Path) -> int | None:
    match = _REEL_STEM_RE.match(reel_path.stem)
    if not match:
        return None
    try:
        index = int(match.group(1))
    except ValueError:
        return None
    return index if index > 0 else None


def sync_reel_markdowns(
    moments: Sequence[Mapping[str, Any]],
    reels_root: Path,
    *,
    max_description_chars: int = DESCRIPTION_MAX_CHARS,
    hashtag_count: int = HASHTAG_COUNT,
) -> list[Path]:
    written: list[Path] = []
    if not reels_root.exists():
        return written

    reel_files = sorted(
        p for p in reels_root.rglob("reel_*.mp4") if p.is_file()
    )
    for reel_path in reel_files:
        index = reel_index_from_path(reel_path)
        if index is None:
            continue
        moment_index = index - 1
        if moment_index < 0 or moment_index >= len(moments):
            continue
        written.append(
            write_reel_markdown(
                moments[moment_index],
                reel_path,
                max_description_chars=max_description_chars,
                hashtag_count=hashtag_count,
            ),
        )
    return written
