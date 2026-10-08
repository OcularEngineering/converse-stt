from __future__ import annotations

import pytest

from ocular_evals.metrics import EditCounts, align, primary_metric, score_text, tokenize
from ocular_evals.text import normalize, punctuation_to_spaces, strip_markup


def test_strip_markup_removes_ocular_tags_and_asides():
    text = "Yeah <breath/> I mean <laugh>totally</laugh> [noise] (aside) <spn/> okay <Clear-throat/>"
    assert strip_markup(text) == "Yeah I mean totally okay"


def test_whisper_profile_applies_english_rules_and_drops_punctuation():
    assert normalize("I'm gonna say twenty-one, okay?") == "i am going to say 21 okay"
    # The Whisper English normalizer drops filler tokens (um, uh, hmm, mm-hmm) on both sides.
    assert normalize("Um, <breath/> right. Mm-hmm.") == "right"
    # Ellipses would otherwise leave an unspeakable "." token behind; currency and percent stay attached.
    assert normalize("She… Yeah. It's... five dollars, 100%.") == "she yeah it is $5 100%"
    assert normalize("I’m here") == "i am here"


def test_plain_profile_keeps_contractions_and_hyphens():
    assert normalize("I’m thinking, well-known <laugh/> stuff!", profile="plain") == "i'm thinking well-known stuff"


def test_non_english_uses_basic_normalizer():
    assert normalize("¡Hola, señor!", language="es") == "hola señor"


def test_unknown_profile_rejected():
    with pytest.raises(ValueError):
        normalize("x", profile="nope")


def test_punctuation_to_spaces_keeps_letters_and_digits():
    assert punctuation_to_spaces("a-b c.d 10%") == "a b c d 10"


def test_tokenize_units():
    assert tokenize("ab cd", "word") == ["ab", "cd"]
    assert tokenize("ab cd", "char") == ["a", "b", "c", "d"]


def test_align_counts_each_operation():
    counts = align("the cat sat on the mat".split(), "the cat sit on the mat today".split())
    assert counts == EditCounts(hits=5, substitutions=1, insertions=1)
    assert align("the cat sat on the mat".split(), "the cat on the mat".split()) == EditCounts(hits=5, deletions=1)
    assert counts.errors == 2
    assert counts.reference_length == 6
    assert counts.hypothesis_length == 7
    assert counts.rate == pytest.approx(2 / 6)


def test_align_handles_empty_sides():
    assert align([], []) == EditCounts()
    assert align([], ["a", "b"]) == EditCounts(insertions=2)
    assert align(["a"], []) == EditCounts(deletions=1)
    assert EditCounts().rate is None


def test_edit_counts_add_and_round_trip():
    total = EditCounts.total([EditCounts(hits=1, deletions=2), EditCounts(hits=3, insertions=1)])
    assert total == EditCounts(hits=4, deletions=2, insertions=1)
    assert EditCounts.from_dict(total.to_dict()) == total
    assert sum([EditCounts(hits=1), EditCounts(hits=2)], EditCounts()) == EditCounts(hits=3)


def test_score_text_gives_wer_and_cer():
    score = score_text("Hello there, <breath/> friend.", "hello their friend")
    assert score.reference == "hello there friend"
    assert score.wer == EditCounts(hits=2, substitutions=1)
    assert score.cer.reference_length == len("hellotherefriend")
    assert score.cer.errors == 2  # "there" -> "their": r/i and e/r swap
    assert score.primary == "wer"
    assert score.to_dict()["wer"]["rate"] == pytest.approx(1 / 3)


def test_primary_metric_by_language():
    assert primary_metric("en-US") == "wer"
    assert primary_metric("zh") == "cer"
    assert primary_metric("ja-JP") == "cer"
