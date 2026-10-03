"""Line layout and text transforms for burned subtitles.

RU: Раскладка реплики по строкам и преобразования текста субтитров.

Line breaks used to be left to libass, which fills the top line greedily and
leaves a one-word tail ("разбираемся, почему этот выпуск / споры"). Cue length
was also counted in characters against a fixed 25-per-line guideline that knew
nothing about the font. This module measures text in the frame's own pixels
with the real font, the way libass will draw it, and picks the line breaks
itself:

* the fewest lines that fit the width (one, when it fits);
* among those, the most even split — or a bottom-heavy pyramid, the shape the
  BBC/Netflix guidelines prefer — at natural points: after punctuation, never
  right after a preposition or conjunction.

The break scoring follows tmoroney/auto-subs (balanced target width, priority
for punctuation) and yochem/cap (penalties for splitting tight word pairs).
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

LINE_BALANCES = ("balanced", "bottom_heavy", "top_heavy", "greedy")
TEXT_CASES = ("none", "upper", "lower", "title")
PUNCTUATION_MODES = ("keep", "periods", "all")
CENSOR_STYLES = ("middle", "whole", "first")

# Average glyph advance as a share of the font size, for when the font file
# cannot be read. Cyrillic display faces sit around 0.33 (condensed) to 0.6
# (wide); 0.45 errs towards shorter lines.
FALLBACK_CHAR_WIDTH_RATIO = 0.45

# Words a line should not end on: a preposition or conjunction left dangling
# reads as an unfinished thought. Shared with the cue splitter.
NO_LINE_END = frozenset({
    "в", "на", "по", "из", "за", "к", "у", "о", "об", "от", "до", "со", "ко",
    "а", "и", "но", "ни", "да", "нет", "не", "то", "ли", "бы", "же", "вот",
    "ну", "или", "что", "как", "где", "когда", "чтобы", "пока", "тоже", "уже",
    "ещё", "еще", "просто", "только", "ведь", "если", "либо", "однако", "потом",
    "тогда", "сейчас", "потому", "раз", "хотя", "чтоб", "будто", "даже",
    "вообще", "именно", "конечно", "пожалуй", "пожалуйста",
    "сразу", "типа", "кроме", "после", "перед", "между", "через", "около",
    "с", "без", "для", "при", "про", "над", "под", "из-за", "из-под",
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "and", "or", "but",
    "with", "from", "by",
})

_SENTENCE_END = re.compile(r"[.!?…]+[»\"')\]]*$")
_CLAUSE_END = re.compile(r"[,;:—–]+[»\"')\]]*$")
_EDGE_PUNCT = ".,!?…;:«»\"'()[]—–-"


def bare_word(word: str) -> str:
    """Lowercased word without surrounding punctuation, for dictionary lookups."""

    return word.strip(_EDGE_PUNCT).lower()


# --- Measuring --------------------------------------------------------------


@lru_cache(maxsize=32)
def _load_font(path: str, ass_size: float):  # type: ignore[no-untyped-def]
    """PIL font sized the way libass sizes ``Fontsize``.

    libass scales a face so that its ascender + descender span the requested
    size, while PIL's size is the em box. Measured against libass 0.17 renders
    the two then agree to a couple of pixels on a 740px line.
    """

    from fontTools.ttLib import TTFont  # noqa: PLC0415 - optional, heavy import
    from PIL import ImageFont  # noqa: PLC0415

    face = TTFont(path, lazy=True)
    upm = int(face["head"].unitsPerEm)
    os2 = face["OS/2"] if "OS/2" in face else None
    span = 0
    if os2 is not None:
        span = int(os2.usWinAscent) + int(os2.usWinDescent)
    if span <= 0:
        hhea = face["hhea"]
        span = int(hhea.ascent) - int(hhea.descent)
    em_size = max(1, round(ass_size * upm / max(span, 1)))
    return ImageFont.truetype(path, em_size)


@dataclass(frozen=True)
class TextMeasurer:
    """Width of a line of text in ASS PlayRes pixels, as libass would draw it."""

    font_path: Path | None
    font_size: float
    scale_x: float = 100.0
    spacing: float = 0.0
    outline: float = 0.0

    def _font(self):  # type: ignore[no-untyped-def]
        if self.font_path is None:
            return None
        try:
            return _load_font(str(self.font_path), float(self.font_size))
        except Exception:  # noqa: BLE001 - missing file, broken font, no PIL
            return None

    def width(self, text: str) -> float:
        font = self._font()
        if font is not None:
            advance = float(font.getlength(text))
        else:
            advance = len(text) * self.font_size * FALLBACK_CHAR_WIDTH_RATIO
        scale = self.scale_x / 100.0
        # \fsp adds spacing after every character, scaled like the glyphs.
        advance += self.spacing * len(text)
        return advance * scale + 2.0 * self.outline

    def average_char_width(self, sample: str) -> float:
        sample = sample or "Разбираемся, почему этот выпуск вызывает споры"
        inner = self.width(sample) - 2.0 * self.outline
        return max(1.0, inner / max(1, len(sample)))


# --- Line breaking ----------------------------------------------------------

# Above this many candidate partitions the exhaustive search falls back to
# greedy filling. Two or three lines of a cue never get close.
_MAX_PARTITIONS = 60_000


def _break_cost(words: Sequence[str], index: int) -> float:
    """Cost of ending a line after ``words[index]`` (lower is better)."""

    word = words[index]
    if _SENTENCE_END.search(word):
        return -0.6
    if _CLAUSE_END.search(word):
        return -0.4
    if bare_word(word) in NO_LINE_END:
        return 0.8
    nxt = words[index + 1] if index + 1 < len(words) else ""
    # A lone short word starting the next line ("и", "а") would rather end
    # this one, but that is the rule above seen from the other side; a short
    # orphan at the very end is handled by the shape cost.
    if nxt and bare_word(nxt) in {"же", "ли", "бы"}:
        return 0.6  # particles cling to the word before them
    return 0.0


def _partition_cost(
    widths: Sequence[float],
    words: Sequence[str],
    breaks: Sequence[int],
    max_width: float,
    balance: str,
    space: float,
) -> float:
    bounds = [0, *breaks, len(words)]
    line_widths = [
        sum(widths[a:b]) + space * max(0, b - a - 1)
        for a, b in zip(bounds, bounds[1:])
    ]
    mean = sum(line_widths) / len(line_widths)
    # Linear, not squared: a natural break point (worth ~0.4-0.6) should win
    # over a slightly more even split, but not over a lopsided one.
    cost = sum(abs(w - mean) for w in line_widths) / max(1.0, max_width)
    for w in line_widths:
        if w > max_width:
            cost += 10.0 + 10.0 * (w - max_width) / max_width
    for b in breaks:
        cost += _break_cost(words, b - 1)
    if balance in ("bottom_heavy", "top_heavy"):
        for upper, lower in zip(line_widths, line_widths[1:]):
            excess = (upper - lower) if balance == "bottom_heavy" else (lower - upper)
            if excess > 0:
                cost += 1.5 * excess / max_width
    # A one-word last line is the classic ugly break.
    if len(bounds) > 2 and bounds[-1] - bounds[-2] == 1 and len(words) > 2:
        cost += 0.5
    return cost


def _greedy(widths: Sequence[float], max_width: float, space: float, max_lines: int) -> list[int]:
    breaks: list[int] = []
    line = 0.0
    for i, w in enumerate(widths):
        if i == 0:
            line = w
            continue
        if line + space + w > max_width and len(breaks) < max_lines - 1:
            breaks.append(i)
            line = w
        else:
            line += space + w
    return breaks


def wrap_words(
    words: Sequence[str],
    measurer: TextMeasurer,
    *,
    max_width: float,
    max_lines: int,
    balance: str = "balanced",
) -> list[list[str]]:
    """Split a cue's words into display lines.

    Uses the fewest lines that fit ``max_width`` (up to ``max_lines``); when
    nothing fits even then, the least-overflowing split of ``max_lines`` lines
    is returned and the caller may split the cue itself.
    """

    words = [w for w in words if w]
    if not words:
        return []
    max_lines = max(1, int(max_lines))
    space = measurer.width(" ") - 2.0 * measurer.outline
    # Outline is paid once per line, not once per word.
    widths = [measurer.width(w) - 2.0 * measurer.outline for w in words]
    max_inner = max(1.0, max_width - 2.0 * measurer.outline)

    total = sum(widths) + space * (len(words) - 1)
    if total <= max_inner or max_lines == 1 or len(words) == 1:
        return [list(words)]

    if balance == "greedy":
        breaks = _greedy(widths, max_inner, space, max_lines)
        return _apply_breaks(words, breaks)

    best: tuple[float, tuple[int, ...]] | None = None
    for lines in range(2, min(max_lines, len(words)) + 1):
        n_partitions = _n_choose_k(len(words) - 1, lines - 1)
        if n_partitions > _MAX_PARTITIONS:
            breaks = _greedy(widths, max_inner, space, lines)
            return _apply_breaks(words, breaks)
        fitting: tuple[float, tuple[int, ...]] | None = None
        for combo in itertools.combinations(range(1, len(words)), lines - 1):
            cost = _partition_cost(widths, words, combo, max_inner, balance, space)
            if fitting is None or cost < fitting[0]:
                fitting = (cost, combo)
        if fitting is None:
            continue
        if best is None or fitting[0] < best[0]:
            best = fitting
        # The first line count with a split that fits wins: more lines than
        # needed would only make the cue taller.
        if fitting[0] < 10.0:
            break
    assert best is not None
    return _apply_breaks(words, best[1])


def lines_fit(lines: Sequence[Sequence[str]], measurer: TextMeasurer, max_width: float) -> bool:
    return all(measurer.width(" ".join(line)) <= max_width + 0.5 for line in lines)


def _apply_breaks(words: Sequence[str], breaks: Iterable[int]) -> list[list[str]]:
    bounds = [0, *breaks, len(words)]
    return [list(words[a:b]) for a, b in zip(bounds, bounds[1:]) if b > a]


def _n_choose_k(n: int, k: int) -> int:
    from math import comb  # noqa: PLC0415

    return comb(n, k) if 0 <= k <= n else 0


# --- Text transforms --------------------------------------------------------


def apply_case(word: str, mode: str) -> str:
    if mode == "upper":
        return word.upper()
    if mode == "lower":
        return word.lower()
    if mode == "title":
        # Capitalise each hyphen part too: «Из-За», «Нью-Йорк».
        return "-".join(part[:1].upper() + part[1:].lower() for part in word.split("-"))
    return word


_TRAILING_SOFT = re.compile(r"(?<![.…])[.,;:]+(?=[»\"')\]]*$)")
_NON_WORD = re.compile(r"[^\w\s'’-]|(?<!\w)[-'’]|[-'’](?!\w)", re.UNICODE)


def strip_punctuation(word: str, mode: str) -> str:
    """``periods``: drop trailing ``. , ; :`` but keep ``?! …`` and ellipses
    (pycaps' default for captions); ``all``: keep letters, digits and
    word-internal hyphens/apostrophes only."""

    if mode == "periods":
        if word.endswith("...") or word.endswith("…"):
            return word
        return _TRAILING_SOFT.sub("", word)
    if mode == "all":
        return _NON_WORD.sub("", word)
    return word


def _normalise(word: str) -> str:
    return "".join(ch for ch in word.lower() if ch.isalnum()).replace("ё", "е")


def build_censor_matcher(entries: Iterable[str]):  # type: ignore[no-untyped-def]
    """Matcher for ``subtitles.censor_words``. A trailing ``*`` makes an entry
    a prefix (``бля*`` covers every form), otherwise the whole word must match."""

    exact: set[str] = set()
    prefixes: list[str] = []
    for raw in entries:
        entry = str(raw).strip()
        if not entry:
            continue
        if entry.endswith("*"):
            stem = _normalise(entry[:-1])
            if stem:
                prefixes.append(stem)
        else:
            norm = _normalise(entry)
            if norm:
                exact.add(norm)

    def matches(word: str) -> bool:
        norm = _normalise(word)
        if not norm:
            return False
        return norm in exact or any(norm.startswith(p) for p in prefixes)

    return matches


def censor_word(word: str, style: str) -> str:
    """Mask the letters of ``word``, leaving punctuation around it intact."""

    match = re.match(r"^(\W*)(.*?)(\W*)$", word, re.UNICODE)
    if not match:
        return word
    lead, core, tail = match.groups()
    if len(core) <= 1:
        return lead + "*" * len(core) + tail
    if style == "whole":
        masked = "*" * len(core)
    elif style == "first":
        masked = core[0] + "*" * (len(core) - 1)
    else:  # middle
        masked = core[0] + "*" * (len(core) - 2) + core[-1] if len(core) > 2 else core[0] + "*"
    return lead + masked + tail
