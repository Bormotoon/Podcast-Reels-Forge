"""RU: Метаданные эпизода из `.info.json`, который оставляет yt-dlp.

Название, описание, главы и теги ролика — то, что автор канала сам написал об
эпизоде. Раньше файл скачивался и не использовался. Теперь из него:

- обзор эпизода получает название, описание и главы с таймкодами (scout видит
  структуру эпизода, а не только выжимку);
- вычитка получает глоссарий: имена и термины в том написании, которое дал
  автор, — ASR на них ошибается чаще всего;
- статья получает настоящее название вместо имени файла.

EN: Episode metadata from the `.info.json` yt-dlp leaves behind.

The video's title, description, chapters and tags are what the channel
author wrote about the episode. The file used to be downloaded and ignored;
now the episode overview gets the title, description and timestamped
chapters, proofreading gets a glossary of names and terms spelled the way the
author spells them (where ASR errs most), and the article gets the real title.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_WORD_RE = re.compile(r"[A-ZА-ЯЁ][\w\-’']{2,}(?:\s+[A-ZА-ЯЁ][\w\-’']{2,})*")
_SENTENCE_START_RE = re.compile(r"(?:^|[.!?…]\s+|\n\s*)$")


@dataclass
class EpisodeMetadata:
    title: str = ""
    description: str = ""
    channel: str = ""
    tags: list[str] = field(default_factory=list)
    #: (start seconds, title)
    chapters: list[tuple[float, str]] = field(default_factory=list)

    def to_analysis_dict(self, *, max_description_chars: int = 1500) -> dict[str, Any]:
        return {
            "title": self.title,
            "channel": self.channel,
            "description": self.description[:max_description_chars],
            "tags": self.tags[:20],
            "chapters": [{"start": start, "title": title} for start, title in self.chapters[:40]],
        }

    def glossary(self, *, max_terms: int = 40) -> list[str]:
        """Names and terms as the author spells them (capitalised mid-sentence, tags)."""

        terms: list[str] = []

        def add(term: str) -> None:
            term = term.strip(" \t\n\"'«»()[]")
            if len(term) >= 3 and term.lower() not in {t.lower() for t in terms}:
                terms.append(term)

        for tag in self.tags:
            if any(ch.isupper() for ch in tag):
                add(tag)
        for text in (self.title, self.description, *(title for _s, title in self.chapters)):
            for match in _WORD_RE.finditer(text or ""):
                if _SENTENCE_START_RE.search(text[: match.start()]) and " " not in match.group(0):
                    continue  # a capital that only starts a sentence
                add(match.group(0))
        return terms[:max_terms]


def info_json_path(source: Path) -> Path:
    return source.with_name(source.stem + ".info.json")


def load_episode_metadata(source: Path) -> EpisodeMetadata | None:
    """Read ``<stem>.info.json`` next to the source, if there is one."""

    path = info_json_path(source)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    chapters: list[tuple[float, str]] = []
    raw_chapters = data.get("chapters")
    if isinstance(raw_chapters, list):
        for chapter in raw_chapters:
            if not isinstance(chapter, dict):
                continue
            try:
                start = float(chapter.get("start_time", 0.0))
            except (TypeError, ValueError):
                continue
            title = str(chapter.get("title") or "").strip()
            if title:
                chapters.append((start, title))

    tags = data.get("tags")
    return EpisodeMetadata(
        title=str(data.get("title") or "").strip(),
        description=str(data.get("description") or "").strip(),
        channel=str(data.get("channel") or data.get("uploader") or "").strip(),
        tags=[str(t).strip() for t in tags if str(t).strip()] if isinstance(tags, list) else [],
        chapters=chapters,
    )
