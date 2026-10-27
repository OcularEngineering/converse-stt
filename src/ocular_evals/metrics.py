"""Word and character error rate built from edit counts.

Every score is stored as hits / substitutions / deletions / insertions rather
than as a percentage. Counts add up across tracks, conversations and the whole
corpus, so the harness never averages percentages of different lengths.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import jiwer

from ocular_evals.text import DEFAULT_PROFILE, language_code, normalize

Unit = Literal["word", "char"]
Metric = Literal["wer", "cer"]

# Languages scored primarily by character error rate because whitespace does
# not mark word boundaries.
CER_LANGUAGES = frozenset({"ja", "ko", "th", "zh", "yue"})


@dataclass(frozen=True)
class EditCounts:
    hits: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def reference_length(self) -> int:
        return self.hits + self.substitutions + self.deletions

    @property
    def hypothesis_length(self) -> int:
        return self.hits + self.substitutions + self.insertions

    @property
    def rate(self) -> float | None:
        if self.reference_length == 0:
            return None
        return self.errors / self.reference_length

    def __add__(self, other: EditCounts) -> EditCounts:
        if not isinstance(other, EditCounts):
            return NotImplemented
        return EditCounts(
            hits=self.hits + other.hits,
            substitutions=self.substitutions + other.substitutions,
            deletions=self.deletions + other.deletions,
            insertions=self.insertions + other.insertions,
        )

    __radd__ = __add__

    def to_dict(self) -> dict[str, int | float | None]:
        data: dict[str, int | float | None] = dict(asdict(self))
        data["errors"] = self.errors
        data["reference_length"] = self.reference_length
        data["rate"] = self.rate
        return data

    @classmethod
    def from_dict(cls, data: dict) -> EditCounts:
        return cls(
            hits=int(data.get("hits", 0)),
            substitutions=int(data.get("substitutions", 0)),
            deletions=int(data.get("deletions", 0)),
            insertions=int(data.get("insertions", 0)),
        )

    @classmethod
    def total(cls, counts) -> EditCounts:
        result = cls()
        for item in counts:
            result = result + item
        return result


class _Identity(jiwer.AbstractTransform):
    """jiwer transform that treats the given list of tokens as one pre-tokenized sentence."""

    def process_string(self, s: str):  # pragma: no cover - jiwer never calls this for lists
        return [[s]]

    def process_list(self, inp):
        return [list(inp)]


_IDENTITY = _Identity()


def tokenize(text: str, unit: Unit) -> list[str]:
    """Split normalized text into scoring tokens: whitespace words or non-space characters."""
    if unit == "word":
        return text.split()
    if unit == "char":
        return [char for char in text if not char.isspace()]
    raise ValueError(f"unknown unit {unit!r}")


def align(reference: list[str], hypothesis: list[str]) -> EditCounts:
    """Count edit operations between two token sequences."""
    if not reference and not hypothesis:
        return EditCounts()
    if not reference:
        return EditCounts(insertions=len(hypothesis))
    if not hypothesis:
        return EditCounts(deletions=len(reference))
    out = jiwer.process_words(
        list(reference),
        list(hypothesis),
        reference_transform=_IDENTITY,
        hypothesis_transform=_IDENTITY,
    )
    return EditCounts(
        hits=out.hits,
        substitutions=out.substitutions,
        deletions=out.deletions,
        insertions=out.insertions,
    )


_OP_CODES = {"equal": "e", "substitute": "s", "delete": "d", "insert": "i"}


def alignment_ops(reference: list[str], hypothesis: list[str]) -> list[tuple[str, int]]:
    """The word alignment behind :func:`align` as runs of ``(op, count)``.

    ``e`` (equal) and ``s`` (substitute) advance both sequences, ``d`` (delete) advances the
    reference only and ``i`` (insert) the hypothesis only. Replaying the runs over the two token
    lists reproduces every edit the score was computed from.
    """
    if not reference and not hypothesis:
        return []
    if not reference:
        return [("i", len(hypothesis))]
    if not hypothesis:
        return [("d", len(reference))]
    out = jiwer.process_words(
        list(reference),
        list(hypothesis),
        reference_transform=_IDENTITY,
        hypothesis_transform=_IDENTITY,
    )
    ops: list[tuple[str, int]] = []
    for chunk in out.alignments[0]:
        if chunk.type == "insert":
            count = chunk.hyp_end_idx - chunk.hyp_start_idx
        else:
            count = chunk.ref_end_idx - chunk.ref_start_idx
        if count:
            ops.append((_OP_CODES[chunk.type], count))
    return ops


def primary_metric(language: str | None) -> Metric:
    return "cer" if language_code(language) in CER_LANGUAGES else "wer"


@dataclass(frozen=True)
class TextScore:
    language: str
    profile: str
    reference: str
    hypothesis: str
    wer: EditCounts
    cer: EditCounts

    @property
    def primary(self) -> Metric:
        return primary_metric(self.language)

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "profile": self.profile,
            "primary_metric": self.primary,
            "reference": self.reference,
            "hypothesis": self.hypothesis,
            "wer": self.wer.to_dict(),
            "cer": self.cer.to_dict(),
        }


def score_text(
    reference: str,
    hypothesis: str,
    *,
    language: str | None = "en",
    profile: str = DEFAULT_PROFILE,
) -> TextScore:
    """Normalize both sides with the same profile and compute WER and CER counts."""
    language = language or "en"
    ref_norm = normalize(reference, language=language, profile=profile)
    hyp_norm = normalize(hypothesis, language=language, profile=profile)
    return TextScore(
        language=language,
        profile=profile,
        reference=ref_norm,
        hypothesis=hyp_norm,
        wer=align(tokenize(ref_norm, "word"), tokenize(hyp_norm, "word")),
        cer=align(tokenize(ref_norm, "char"), tokenize(hyp_norm, "char")),
    )


def percent(rate: float | None, digits: int = 2) -> str:
    if rate is None:
        return "n/a"
    return f"{rate * 100:.{digits}f}%"
