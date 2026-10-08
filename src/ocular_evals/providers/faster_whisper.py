"""Local Whisper inference through faster-whisper / CTranslate2 (optional ``local`` extra)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ocular_evals.config import ModelConfig
from ocular_evals.providers.base import Hypothesis, Provider, ProviderError, primary_language_tag

INSTALL_HINT = "faster-whisper is not installed; install the `local` extra: uv sync --extra local"
DEVICES = ("auto", "cpu", "cuda")


def _import_whisper_model():
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ProviderError(INSTALL_HINT) from exc
    return WhisperModel


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class FasterWhisperProvider(Provider):
    key = "faster-whisper"
    option_names = frozenset(
        {
            "device",
            "compute_type",
            "cpu_threads",
            "beam_size",
            "best_of",
            "temperature",
            "condition_on_previous_text",
            "vad_filter",
            "vad_min_silence_ms",
            "hallucination_silence_threshold",
            "word_timestamps",
            "download_root",
            "revision",
            "language_hint",
        }
    )

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        options = config.options
        self.device = str(options.get("device", "auto"))
        self.compute_type = str(options.get("compute_type", "default"))
        self.cpu_threads = options.get("cpu_threads")
        self.beam_size = options.get("beam_size", 5)
        self.best_of = options.get("best_of")
        self.temperature = options.get("temperature", 0.0)
        self.condition_on_previous_text = options.get("condition_on_previous_text", False)
        self.vad_filter = options.get("vad_filter", True)
        self.vad_min_silence_ms = options.get("vad_min_silence_ms")
        self.hallucination_silence_threshold = options.get("hallucination_silence_threshold")
        self.word_timestamps = options.get("word_timestamps", False)
        self.download_root = options.get("download_root")
        self.revision = options.get("revision")
        self.language_hint = options.get("language_hint", True)
        self._model: Any | None = None

    def check(self) -> list[str]:
        problems = super().check()
        try:
            _import_whisper_model()
        except ProviderError as exc:
            problems.append(str(exc))
        if self.device not in DEVICES:
            problems.append(f"device must be one of {', '.join(DEVICES)}; got {self.device!r}")
        if self.cpu_threads is not None and (
            isinstance(self.cpu_threads, bool) or not isinstance(self.cpu_threads, int) or self.cpu_threads < 1
        ):
            problems.append("cpu_threads must be a positive integer")
        if isinstance(self.beam_size, bool) or not isinstance(self.beam_size, int) or self.beam_size < 1:
            problems.append("beam_size must be a positive integer")
        if self.best_of is not None and (
            isinstance(self.best_of, bool) or not isinstance(self.best_of, int) or self.best_of < 1
        ):
            problems.append("best_of must be a positive integer")
        temperatures = self.temperature if isinstance(self.temperature, list) else [self.temperature]
        if not temperatures or not all(_is_number(value) and value >= 0 for value in temperatures):
            problems.append("temperature must be a non-negative number or a list of them")
        for name in ("condition_on_previous_text", "vad_filter", "word_timestamps", "language_hint"):
            if not isinstance(getattr(self, name), bool):
                problems.append(f"{name} must be true or false")
        if self.vad_min_silence_ms is not None and (
            isinstance(self.vad_min_silence_ms, bool)
            or not isinstance(self.vad_min_silence_ms, int)
            or self.vad_min_silence_ms < 0
        ):
            problems.append("vad_min_silence_ms must be a non-negative integer")
        if self.hallucination_silence_threshold is not None and not (
            _is_number(self.hallucination_silence_threshold) and self.hallucination_silence_threshold >= 0
        ):
            problems.append("hallucination_silence_threshold must be a non-negative number")
        if self.download_root is not None and not isinstance(self.download_root, str):
            problems.append("download_root must be a path string")
        if self.revision is not None and not isinstance(self.revision, str):
            problems.append("revision must be a string")
        return problems

    def _load(self):
        if self._model is not None:
            return self._model
        WhisperModel = _import_whisper_model()
        kwargs: dict[str, Any] = {"device": self.device, "compute_type": self.compute_type}
        if self.cpu_threads is not None:
            kwargs["cpu_threads"] = int(self.cpu_threads)
        if self.download_root is not None:
            kwargs["download_root"] = str(self.download_root)
        try:
            if self.revision is not None:
                try:
                    self._model = WhisperModel(self.config.model, revision=self.revision, **kwargs)
                except TypeError:
                    self._model = WhisperModel(self.config.model, **kwargs)
            else:
                self._model = WhisperModel(self.config.model, **kwargs)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"faster-whisper could not load {self.config.model!r}: {exc}") from exc
        return self._model

    def _decode_options(self, language: str) -> dict[str, Any]:
        options: dict[str, Any] = {
            "beam_size": int(self.beam_size),
            "temperature": self.temperature,
            "condition_on_previous_text": bool(self.condition_on_previous_text),
            "vad_filter": bool(self.vad_filter),
            "word_timestamps": bool(self.word_timestamps),
        }
        if self.language_hint:
            options["language"] = primary_language_tag(language)
        if self.best_of is not None:
            options["best_of"] = int(self.best_of)
        if self.vad_min_silence_ms is not None:
            options["vad_parameters"] = {"min_silence_duration_ms": int(self.vad_min_silence_ms)}
        if self.hallucination_silence_threshold is not None:
            options["hallucination_silence_threshold"] = float(self.hallucination_silence_threshold)
        return options

    def transcribe(self, audio: Path, *, language: str) -> Hypothesis:
        model = self._load()
        options = self._decode_options(language)
        try:
            generator, info = model.transcribe(str(audio), **options)
            segments: list[dict] = []
            texts: list[str] = []
            for segment in generator:
                text = str(segment.text or "").strip()
                if text:
                    texts.append(text)
                segments.append({"start_s": float(segment.start), "end_s": float(segment.end), "text": text})
        except Exception as exc:
            raise ProviderError(f"faster-whisper failed on {audio.name}: {exc}") from exc
        probability = getattr(info, "language_probability", None)
        duration = getattr(info, "duration", None)
        return Hypothesis(
            text=" ".join(texts),
            language=getattr(info, "language", None),
            duration_s=float(duration) if duration is not None else None,
            segments=tuple(segments),
            raw={
                "model": self.config.model,
                "device": self.device,
                "compute_type": self.compute_type,
                "revision": self.revision,
                "decode_options": options,
                "language_probability": float(probability) if probability is not None else None,
            },
        )

    def close(self) -> None:
        self._model = None


__all__ = ["FasterWhisperProvider"]
