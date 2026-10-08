"""Provider contract shared by every speech-to-text backend."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from ocular_evals.config import ModelConfig
from ocular_evals.env import redact


@dataclass(frozen=True)
class Hypothesis:
    """One transcript for one audio file."""

    text: str
    language: str | None = None
    duration_s: float | None = None
    segments: tuple[dict, ...] = ()
    raw: dict = field(default_factory=dict)


class ProviderError(RuntimeError):
    """A provider could not produce a hypothesis. Messages are redacted before they are stored."""

    def __init__(self, message: str, *, retryable: bool = False, status: int | None = None) -> None:
        super().__init__(redact(message))
        self.retryable = retryable
        self.status = status


class Provider(ABC):
    """Turn an audio file into a :class:`Hypothesis`."""

    key: ClassVar[str]
    option_names: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, config: ModelConfig) -> None:
        self._config = config

    @property
    def config(self) -> ModelConfig:
        return self._config

    def required_env(self) -> tuple[str, ...]:
        """Environment variables the provider itself needs (in addition to ``config.env``)."""
        return ()

    def check(self) -> list[str]:
        """Return the problems that would prevent a run (missing env vars, unknown options). No network."""
        problems: list[str] = []
        names = list(self._config.env) + [name for name in self.required_env() if name not in self._config.env]
        problems.extend(missing_env(names))
        problems.extend(unknown_options(self._config.options, self.option_names))
        return problems

    @abstractmethod
    def transcribe(self, audio: Path, *, language: str) -> Hypothesis:
        """Transcribe ``audio``; ``language`` is a BCP-47 primary tag such as ``"en"``."""

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def missing_env(names: Iterable[str]) -> list[str]:
    return [f"environment variable {name} is not set" for name in names if not os.environ.get(name)]


def unknown_options(options: Mapping[str, Any], allowed: Iterable[str]) -> list[str]:
    allowed_set = set(allowed)
    unknown = sorted(name for name in options if name not in allowed_set)
    if not unknown:
        return []
    return [f"unknown option(s): {', '.join(unknown)}; allowed: {', '.join(sorted(allowed_set))}"]


def primary_language_tag(language: str) -> str:
    """``"en-US"`` -> ``"en"``."""
    return language.split("-", 1)[0].strip().lower()


def coerce_segments(items: object) -> tuple[dict, ...]:
    """Map provider ``{"start", "end", "text"}`` items onto ``{"start_s", "end_s", "text"}`` dicts."""
    if not isinstance(items, list):
        return ()
    shaped: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        start = item.get("start_s", item.get("start"))
        end = item.get("end_s", item.get("end"))
        shaped.append(
            {
                "start_s": float(start) if isinstance(start, (int, float)) and not isinstance(start, bool) else None,
                "end_s": float(end) if isinstance(end, (int, float)) and not isinstance(end, bool) else None,
                "text": str(item.get("text") or "").strip(),
            }
        )
    return tuple(shaped)


__all__ = [
    "Hypothesis",
    "Provider",
    "ProviderError",
    "coerce_segments",
    "missing_env",
    "primary_language_tag",
    "unknown_options",
]
