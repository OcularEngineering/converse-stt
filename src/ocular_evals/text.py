"""Text normalization shared by references and hypotheses.

Two profiles are available:

``whisper``
    Strip Ocular transcript markup, then apply the OpenAI Whisper text
    normalizer (English rules for English, the basic Unicode rules otherwise),
    then drop any token without a letter or digit. This is the profile used
    for the numbers on the Converse-STT board and matches the public research
    harnesses we compare against.

``plain``
    Strip markup, lowercase, fold curly apostrophes, drop punctuation except
    apostrophes and hyphens inside words. Useful when you want to see errors
    without the normalizer rewriting contractions and numbers.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

PROFILES: tuple[str, ...] = ("whisper", "plain")
DEFAULT_PROFILE = "whisper"

# Ocular transcripts annotate non-speech with SGML style tags such as
# ``<breath/>``, ``<laugh>...</laugh>``, ``<spn/>`` and ``<unclear/>``.
_TAG = re.compile(r"</?[^<>]{0,120}>")
_BRACKET = re.compile(r"\[[^\[\]]{0,120}\]")
_PAREN = re.compile(r"\([^()]*\)")
_WHITESPACE = re.compile(r"\s+")
_HAS_ALNUM = re.compile(r"[^\W_]", re.UNICODE)
_PLAIN_DROP = re.compile(r"[^\w\s'-]", re.UNICODE)
_PLAIN_EDGE_PUNCT = re.compile(r"(?<![\w])['-]+|['-]+(?![\w])", re.UNICODE)
_APOSTROPHES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'", "`": "'", "\u00b4": "'"})


def language_code(language: str | None) -> str:
    """Return the lowercase primary subtag of a BCP 47 tag (``en-US`` -> ``en``)."""
    if not language:
        return "en"
    return language.replace("_", "-").split("-")[0].strip().lower() or "en"


def collapse_whitespace(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def strip_markup(text: str) -> str:
    """Remove transcript annotations: tags, bracketed notes and parenthetical asides."""
    text = _TAG.sub(" ", text)
    text = _BRACKET.sub(" ", text)
    previous = None
    while previous != text:
        previous = text
        text = _PAREN.sub(" ", text)
    text = text.replace("(", " ").replace(")", " ")
    return collapse_whitespace(text)


def punctuation_to_spaces(text: str) -> str:
    """Replace Unicode punctuation and symbols with spaces, keeping combining marks."""
    out: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        if category[0] in {"P", "S"}:
            out.append(" ")
        else:
            out.append(char)
    return collapse_whitespace("".join(out))


@lru_cache(maxsize=1)
def _english_normalizer():
    from whisper_normalizer.english import EnglishTextNormalizer

    return EnglishTextNormalizer()


@lru_cache(maxsize=1)
def _basic_normalizer():
    from whisper_normalizer.basic import BasicTextNormalizer

    return BasicTextNormalizer()


def _normalize_plain(text: str) -> str:
    text = text.translate(_APOSTROPHES).lower()
    text = _PLAIN_DROP.sub(" ", text)
    text = _PLAIN_EDGE_PUNCT.sub(" ", text)
    return collapse_whitespace(text)


def _normalize_whisper(text: str, language: str) -> str:
    text = text.translate(_APOSTROPHES)
    if language_code(language) == "en":
        # Keeps Whisper's number formatting ("$5", "100%", "7.0") so both sides agree on it.
        text = _english_normalizer()(text)
    else:
        text = _basic_normalizer()(text)
    # A token with no letter or digit (a stray "." left after a filler was removed) cannot be spoken.
    return " ".join(token for token in text.split() if _HAS_ALNUM.search(token)).lower()


def normalize(text: str | None, *, language: str | None = "en", profile: str = DEFAULT_PROFILE) -> str:
    """Normalize ``text`` for scoring.

    The same function is applied to the reference and the hypothesis so that
    only genuine recognition differences are counted.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown normalization profile {profile!r}; expected one of {', '.join(PROFILES)}")
    text = unicodedata.normalize("NFKC", text or "")
    text = strip_markup(text)
    if not text:
        return ""
    if profile == "plain":
        return _normalize_plain(text)
    return _normalize_whisper(text, language or "en")
