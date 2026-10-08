"""Ocular Converse-STT evaluation harness.

Transcribe each speaker track of an Ocular full-duplex conversation with a
speech-to-text model, normalize reference and hypothesis the same way, and
report word error rate (WER) and character error rate (CER).
"""

from ocular_evals.metrics import EditCounts, TextScore, score_text
from ocular_evals.text import PROFILES, normalize, strip_markup

__version__ = "0.1.0"

__all__ = [
    "EditCounts",
    "PROFILES",
    "TextScore",
    "__version__",
    "normalize",
    "score_text",
    "strip_markup",
]
