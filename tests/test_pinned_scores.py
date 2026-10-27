"""Pinned scores: the published boards depend on these exact numbers.

Converse-STT publishes a board version per scoring methodology. Anything that changes what
``score_text`` returns for the passage below (the normalizer and its pinned ``whisper-normalizer``
release, markup stripping, tokenization, the alignment) changes every published number and is a
**new board version**, not a fix to the current one.

If this test fails on purpose: bump the board version in ``src/ocular_evals/boards/converse-stt.json``,
bump the package version, re-pin the counts here in the same commit, and say so in the changelog. The
site syncs refuse an export whose harness version differs from the committed board unless told it is
a new version, so the pin here is the first of two gates.
"""

from __future__ import annotations

import pytest

from ocular_evals.metrics import score_text

# Conversational English with the things that make normalization matter: fillers, a non-speech tag,
# a bracketed aside, numbers as words and digits, currency, a title, an acronym, contractions and a
# repeated word at the end of the hypothesis.
REFERENCE = (
    "Yeah, so <breath/> I've been thinking about it, um, since like twenty twenty-three. "
    "We spent $1,200 on the first drone and, uh, it didn't even fly for 10 minutes. [laughs] "
    "Right, right. Mr. Alvarez said the FAA rules were gonna change in November."
)
HYPOTHESIS = (
    "yeah so I been thinking about it since like 2023 "
    "we spent twelve hundred dollars on the first drone and it didn't even fly for ten minutes "
    "right right mister alvarez said the F.A.A. rules are going to change in november november"
)

NORMALIZED_REFERENCE = (
    "yeah so i have been thinking about it since like 2023 we spent $1200 on the 1st drone and it did not "
    "even fly for 10 minutes right right mister alvarez said the faa rules were going to change in november"
)
NORMALIZED_HYPOTHESIS = (
    "yeah so i been thinking about it since like 2023 we spent $1200 on the 1st drone and it did not even "
    "fly for 10 minutes right right mister alvarez said the f a a rules are going to change in november november"
)
WER_COUNTS = {"hits": 38, "substitutions": 2, "deletions": 1, "insertions": 3, "errors": 6, "reference_length": 41}
CER_COUNTS = {"hits": 157, "substitutions": 1, "deletions": 5, "insertions": 8, "errors": 14, "reference_length": 163}


def test_whisper_profile_scores_are_pinned_to_the_published_board():
    score = score_text(REFERENCE, HYPOTHESIS, language="en", profile="whisper")
    assert score.reference == NORMALIZED_REFERENCE, "normalizer output changed: new board version"
    assert score.hypothesis == NORMALIZED_HYPOTHESIS, "normalizer output changed: new board version"
    wer, cer = score.wer.to_dict(), score.cer.to_dict()
    assert {k: wer[k] for k in WER_COUNTS} == WER_COUNTS, "word alignment changed: new board version"
    assert {k: cer[k] for k in CER_COUNTS} == CER_COUNTS, "character alignment changed: new board version"
    assert wer["rate"] == pytest.approx(6 / 41) and cer["rate"] == pytest.approx(14 / 163)
