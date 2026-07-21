"""RU: Перепроверка редких слов и имён собственных во внешних источниках.

Распознавание речи стабильно ошибается на именах и прозвищах: «Курокрад»
превращается в «Курократ». Внутри эпизода это не поймать — правильной формы
там просто нет, — а дёргать человека на каждое слово незачем.

Идея: не спрашивать «как правильно», а проверять, какое из написаний вообще
известно внешнему источнику. У «курокрад» в Викисловаре девять совпадений,
у «курократ» — ноль; этого достаточно, чтобы выбрать.

Правило безопасности: правка вносится, только когда исходное написание
источнику неизвестно, а вариант известен уверенно. Во всех остальных случаях
(оба известны, оба неизвестны, сети нет) текст остаётся как есть — выдумывать
написание хуже, чем оставить ошибку распознавания.

EN: Re-checking rare words and proper nouns against external sources.

Speech recognition reliably trips over names and nicknames: "Курокрад" comes
back as "Курократ". The episode itself cannot settle it — the correct form is
simply absent — and asking a human about every word defeats the point.

The idea is not to ask "what is correct" but to check which spelling an outside
source knows at all. Wiktionary has nine hits for "курокрад" and none for
"курократ", which is enough to choose.

Safety rule: a fix is applied only when the original spelling is unknown to the
source and a variant is known with confidence. In every other case — both known,
neither known, no network — the text is left alone, because inventing a spelling
is worse than keeping a recognition error.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

# Consonant pairs ASR confuses at the end of a word (Russian devoices them), and
# the vowel pairs it confuses when unstressed. These cover the bulk of the
# damage: "курокраД" -> "курокраТ", "ДЭнавалис" -> "ДЕнавалис".
_CONFUSABLE_PAIRS = (
    ("д", "т"), ("б", "п"), ("в", "ф"), ("г", "к"), ("ж", "ш"), ("з", "с"),
    ("е", "э"), ("и", "ы"), ("о", "а"), ("ш", "щ"), ("ъ", "ь"),
)

_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
# A capital letter that is not opening a sentence: the cheapest proper-noun hint
# there is, and no dictionary needed.
_MID_SENTENCE_CAPITAL_RE = re.compile(
    r"(?<![.!?…\n]\s)(?<!^)\b([А-ЯЁA-Z][а-яёa-z]{3,})\b",
    re.MULTILINE,
)

#: Terms sharing this much of a prefix are treated as forms of one word.
_STEM_LEN = 6
#: Shorter terms are skipped. An inflected form of an ordinary name ("Жене") is
#: absent from a dictionary just like a genuine mis-hearing is, but it is not an
#: error — and a short word has near-homophones everywhere. Length is the
#: cheapest way to tell the two apart.
_MIN_TERM_LEN = 6
#: Below this many hits a spelling counts as unknown to the source.
DEFAULT_UNKNOWN_HITS = 1
#: A replacement needs at least this many hits to be trusted.
DEFAULT_CONFIDENT_HITS = 5


@dataclass(frozen=True)
class TermFix:
    """One spelling correction backed by evidence."""

    wrong: str
    right: str
    occurrences: int
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "wrong": self.wrong,
            "right": self.right,
            "occurrences": self.occurrences,
            "evidence": self.evidence,
        }


class TermVerifier(Protocol):
    """Something that can say how well an external source knows a spelling.

    Returns ``None`` when it could not find out — a failed lookup must never be
    mistaken for "this spelling is unknown", or a flaky network would start
    rewriting words.
    """

    def hits(self, term: str, context: str = "") -> int | None: ...


@dataclass(frozen=True)
class SuspectTerm:
    """A term worth checking, with the words that give it context."""

    term: str
    occurrences: int
    context: str = ""

    def query(self) -> str:
        return f"{self.context} {self.term}".strip()


def find_suspect_terms(text: str, *, min_occurrences: int = 2) -> list[SuspectTerm]:
    """RU: Находит имена собственные и редкие слова, стоящие проверки.

    EN: Find the proper nouns and rare words worth checking.

    Inflected forms are folded together by stem and the shortest is reported —
    usually the nominative, which is what an external source has an entry for.

    Each term also carries its most frequent capitalized neighbour. On its own a
    rare name may be unfindable while the phrase around it is not: the club in
    one episode is "Подземелье Деновалис", and the second word alone leads
    nowhere.
    """
    body = text or ""
    candidates = _MID_SENTENCE_CAPITAL_RE.findall(body)
    if not candidates:
        return []

    groups: dict[str, list[str]] = {}
    for word in candidates:
        groups.setdefault(word.lower()[:_STEM_LEN], []).append(word)

    out: list[SuspectTerm] = []
    for forms in groups.values():
        if len(forms) < min_occurrences:
            continue
        if min(len(form) for form in forms) < _MIN_TERM_LEN:
            continue
        # The shortest form is the closest thing to a dictionary headword.
        base = min(forms, key=len)
        out.append(
            SuspectTerm(
                term=base,
                occurrences=len(forms),
                context=_collocate(body, forms),
            ),
        )
    out.sort(key=lambda item: -item.occurrences)
    return out


def _collocate(text: str, forms: Sequence[str]) -> str:
    """The capitalized word most often standing next to these forms."""

    counts: dict[str, int] = {}
    for form in set(forms):
        for match in re.finditer(
            r"\b([А-ЯЁA-Z][а-яёa-z]{3,})\s+" + re.escape(form) + r"\b", text,
        ):
            neighbour = match.group(1)
            if neighbour.lower()[:_STEM_LEN] == form.lower()[:_STEM_LEN]:
                continue
            counts[neighbour] = counts.get(neighbour, 0) + 1
    if not counts:
        return ""
    best = max(counts.items(), key=lambda kv: kv[1])
    return best[0] if best[1] >= 1 else ""


def spelling_variants(term: str, *, max_variants: int = 24) -> list[str]:
    """RU: Правдоподобные варианты написания слова.

    EN: Plausible alternative spellings of a term.

    Only single-letter swaps between sounds speech recognition mixes up, so the
    list stays short and every entry is a spelling a human might actually write.
    """
    lowered = (term or "").lower()
    if not lowered:
        return []

    swaps: dict[str, str] = {}
    for left, right in _CONFUSABLE_PAIRS:
        swaps.setdefault(left, right)
        swaps.setdefault(right, left)

    seen: set[str] = {lowered}
    variants: list[str] = []
    # Walk from the end: the last consonant is where devoicing bites. Position 0
    # is deliberately excluded — speech recognition rarely loses the first
    # sound, and swapping it just yields a different word ("Жене" -> "Шене").
    for index in range(len(lowered) - 1, 0, -1):
        replacement = swaps.get(lowered[index])
        if not replacement:
            continue
        candidate = lowered[:index] + replacement + lowered[index + 1:]
        if candidate in seen:
            continue
        seen.add(candidate)
        variants.append(_match_case(term, candidate))
        if len(variants) >= max_variants:
            break
    return variants


def _match_case(model: str, value: str) -> str:
    """Give *value* the capitalization pattern of *model*."""

    if model[:1].isupper():
        return value[:1].upper() + value[1:]
    return value


class WikiTermVerifier:
    """RU: Проверка написания через официальный API MediaWiki.

    EN: Spelling check through the official MediaWiki API.

    Wiktionary first (it has entries for ordinary and slang words), Wikipedia
    second (notable names). Both are public, keyless and meant to be queried —
    unlike scraping a search engine's result page.

    Only the bare term is ever sent: no transcript text leaves the machine.
    """

    def __init__(
        self,
        *,
        sites: Sequence[str] = ("ru.wiktionary.org", "ru.wikipedia.org"),
        timeout: float = 10.0,
        pause: float = 1.0,
        user_agent: str = "PodcastReelsForge/1.0 (term spell-check)",
    ) -> None:
        self.sites = tuple(sites)
        self.timeout = float(timeout)
        self.pause = float(pause)
        self.user_agent = user_agent

    def hits(self, term: str, context: str = "") -> int | None:
        # A dictionary is looked up by headword, so the context is deliberately
        # dropped here: "Подземелье курокрад" has no entry, "курокрад" does.
        _ = context
        total = 0
        for site in self.sites:
            found = self._site_hits(site, term)
            if found is None:
                return None
            total += found
            if total:
                break
        return total

    def _site_hits(self, site: str, term: str) -> int | None:
        query = urllib.parse.urlencode({
            "action": "query",
            "list": "search",
            "srsearch": term,
            "srlimit": 1,
            "format": "json",
        })
        url = f"https://{site}/w/api.php?{query}"
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            # No verdict rather than "zero hits": see TermVerifier.
            return None
        finally:
            if self.pause:
                time.sleep(self.pause)

        info = payload.get("query", {}).get("searchinfo", {})
        try:
            return int(info.get("totalhits", 0))
        except (TypeError, ValueError):
            return 0


def verify_terms(
    terms: Sequence[SuspectTerm],
    verifier: TermVerifier,
    *,
    cache: dict[str, int] | None = None,
    unknown_hits: int = DEFAULT_UNKNOWN_HITS,
    confident_hits: int = DEFAULT_CONFIDENT_HITS,
    max_terms: int = 40,
) -> list[TermFix]:
    """RU: Подбирает исправления для написаний, неизвестных источнику.

    EN: Work out corrections for spellings the source does not know.

    A term the source already knows costs exactly one lookup and is left alone;
    only an unknown one is worth generating variants for. A variant wins only by
    a clear margin, and if none does, the original stays — the point is to catch
    obvious recognition errors, not to overwrite anything unusual.
    """
    lookups = dict(cache) if cache is not None else {}

    def hits(term: str, context: str) -> int | None:
        key = f"{context}|{term}".lower().strip("|")
        if key not in lookups:
            found = verifier.hits(term, context)
            if found is None:
                return None
            lookups[key] = int(found)
        return lookups[key]

    fixes: list[TermFix] = []
    for suspect in list(terms)[:max_terms]:
        original = hits(suspect.term, suspect.context)
        if original is None or original >= unknown_hits:
            # Either the source knows this spelling, or it could not be asked.
            continue

        best: tuple[str, int] | None = None
        for variant in spelling_variants(suspect.term):
            found = hits(variant, suspect.context)
            if found is None:
                best = None
                break
            if found >= confident_hits and (best is None or found > best[1]):
                best = (variant, found)

        if best is not None:
            where = f" (context: {suspect.context})" if suspect.context else ""
            fixes.append(
                TermFix(
                    wrong=suspect.term,
                    right=best[0],
                    occurrences=suspect.occurrences,
                    evidence=f"{suspect.term}: 0 hits, {best[0]}: {best[1]} hits{where}",
                ),
            )

    if cache is not None:
        cache.update(lookups)
    return fixes


def apply_term_fixes(text: str, fixes: Iterable[TermFix]) -> tuple[str, int]:
    """RU: Применяет исправления ко всем формам слова, сохраняя окончания.

    EN: Apply the fixes to every inflected form, keeping the endings.

    "Курократа" becomes "Курокрада": the stem is swapped and whatever the word
    ended with is carried over untouched.
    """
    out = text or ""
    replaced = 0

    for fix in fixes:
        pattern = re.compile(
            r"\b(" + re.escape(fix.wrong) + r")([а-яёa-z]*)\b",
            re.IGNORECASE | re.UNICODE,
        )

        def substitute(match: re.Match[str], fix: TermFix = fix) -> str:
            head, tail = match.group(1), match.group(2)
            replacement = _match_case(head, fix.right.lower())
            return replacement + tail

        out, count = pattern.subn(substitute, out)
        replaced += count

    return out, replaced


def load_cache(path: Path | None) -> dict[str, int]:
    """Read the lookup cache, so a term is only ever queried once."""

    if not path or not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, int] = {}
    for key, value in raw.items():
        try:
            out[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return out


def save_cache(path: Path | None, cache: Mapping[str, int]) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(sorted(cache.items())), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
