"""Tests for re-checking rare spellings against an external source."""

from __future__ import annotations

from pathlib import Path

from podcast_reels_forge.analysis.term_check import (
    TermFix,
    apply_term_fixes,
    find_suspect_terms,
    load_cache,
    save_cache,
    spelling_variants,
    verify_terms,
)


class FakeVerifier:
    """Stands in for the web: a fixed table of what the source 'knows'."""

    def __init__(self, known: dict[str, int], *, fail: set[str] | None = None):
        self.known = {k.lower(): v for k, v in known.items()}
        self.fail = {f.lower() for f in (fail or set())}
        self.queries: list[tuple[str, str]] = []

    def hits(self, term: str, context: str = "") -> int | None:
        self.queries.append((term, context))
        if term.lower() in self.fail:
            return None
        return self.known.get(term.lower(), 0)


def test_variants_cover_the_devoicing_confusions() -> None:
    """The error this exists for: a final д heard as т."""
    assert "Курокрад" in spelling_variants("Курократ")
    # And the vowel confusion in the middle of a name.
    assert "Деновалис" in spelling_variants("Денавалис")


def test_variants_never_touch_the_first_letter() -> None:
    """Swapping the first sound yields a different word, not a mis-hearing."""
    for variant in spelling_variants("Жеребец"):
        assert variant[0].lower() == "ж", variant


def test_find_suspect_terms_groups_inflections_and_keeps_context() -> None:
    text = (
        "Сегодня со мной Женю Курократа позвали. "
        "Я спросил Курократа про клуб. "
        "И вот Женю Курократ рассказывает."
    )

    suspects = find_suspect_terms(text, min_occurrences=2)
    by_term = {s.term: s for s in suspects}

    assert "Курократ" in by_term
    assert by_term["Курократ"].occurrences == 3
    # A bare rare name can be unfindable while the phrase around it is not.
    assert by_term["Курократ"].context == "Женю"
    assert by_term["Курократ"].query() == "Женю Курократ"


def test_short_terms_are_left_alone() -> None:
    """"Жене" is a case form of an ordinary name, not a mis-hearing."""
    text = "Я сказал Жене. Потом снова Жене. И ещё раз Жене."

    assert [s.term for s in find_suspect_terms(text, min_occurrences=2)] == []


def test_verify_terms_fixes_only_what_the_source_settles() -> None:
    suspects = find_suspect_terms(
        "Позвали Женю Курократа. Спросили Курократа. Пришёл Женю Курократ.",
        min_occurrences=2,
    )
    verifier = FakeVerifier({"Курокрад": 9})

    fixes = verify_terms(suspects, verifier)

    assert len(fixes) == 1
    assert (fixes[0].wrong, fixes[0].right) == ("Курократ", "Курокрад")
    assert "9 hits" in fixes[0].evidence


def test_a_spelling_the_source_knows_is_never_touched() -> None:
    # The nominative is present, so that is the form looked up — the base form
    # is the shortest one actually seen, which is as close to a headword as the
    # transcript gets.
    suspects = find_suspect_terms(
        "Это Зеленоград. Дошли до Зеленограда.", min_occurrences=2,
    )
    assert [s.term for s in suspects] == ["Зеленоград"]

    # The source knows the original, so no variant is even considered.
    verifier = FakeVerifier({"Зеленоград": 500, "Зеленокрад": 900})

    assert verify_terms(suspects, verifier) == []
    assert len(verifier.queries) == 1, "a known term costs exactly one lookup"


def test_nothing_is_changed_when_no_variant_is_known() -> None:
    """A local club nobody indexed must survive untouched."""
    suspects = find_suspect_terms(
        "Клуб Подземелье Денавалис. Снова Денавалис.", min_occurrences=2,
    )

    assert verify_terms(suspects, FakeVerifier({})) == []


def test_a_failed_lookup_is_not_read_as_unknown() -> None:
    """A flaky network must not start rewriting words."""
    suspects = find_suspect_terms(
        "Позвали Женю Курократа. Спросили Курократа.", min_occurrences=2,
    )
    verifier = FakeVerifier({"Курокрад": 9}, fail={"Курократ"})

    assert verify_terms(suspects, verifier) == []


def test_apply_fixes_keeps_endings_and_case() -> None:
    text = "Пришёл Курократ. Позвали Курократа. Дали курократу денег."

    fixed, count = apply_term_fixes(text, [TermFix("Курократ", "Курокрад", 3, "")])

    assert fixed == "Пришёл Курокрад. Позвали Курокрада. Дали курокраду денег."
    assert count == 3


def test_cache_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "terms.json"
    save_cache(path, {"курокрад": 9, "курократ": 0})

    assert load_cache(path) == {"курокрад": 9, "курократ": 0}
    assert load_cache(tmp_path / "missing.json") == {}
    assert load_cache(None) == {}


def test_cache_spares_repeat_lookups() -> None:
    suspects = find_suspect_terms(
        "Позвали Женю Курократа. Спросили Курократа.", min_occurrences=2,
    )
    verifier = FakeVerifier({"Курокрад": 9})
    cache: dict[str, int] = {}

    verify_terms(suspects, verifier, cache=cache)
    first_round = len(verifier.queries)
    verify_terms(suspects, verifier, cache=cache)

    assert len(verifier.queries) == first_round, "the second run must be free"
