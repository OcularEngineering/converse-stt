"""Local inference for NVIDIA's Nemotron cache-aware streaming ASR models via 🤗 Transformers (``nemotron`` extra).

Covers ``nvidia/nemotron-speech-streaming-en-0.6b`` (Nemotron 3 ASR, English) and
``nvidia/nemotron-3.5-asr-streaming-0.6b`` (Nemotron 3.5 ASR, multilingual). Both are
FastConformer-RNNT encoders trained for cache-aware streaming: the right attention context
(``lookahead_frames``, in 80 ms encoder frames) fixes the chunk size the model "hears" ahead of
each frame, and because streaming steps never overlap, offline inference at a given lookahead
produces the same transcript the streaming path would at that chunk size. The lookahead is
therefore the one knob that moves these models between their lowest-latency (80 ms) and most
accurate (1.12 s) operating points, and it is recorded in the hypothesis for the audit trail.

The Transformers implementation caps one request at ``max_position_embeddings`` encoder frames
(5000 = 400 s), so longer files are cut into ``max_chunk_s`` pieces at quiet points with the same
helper the hosted providers use, transcribed in order, and joined.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from ocular_evals.config import ModelConfig
from ocular_evals.providers.audio import audio_info, split_wav16k
from ocular_evals.providers.base import Hypothesis, Provider, ProviderError

INSTALL_HINT = "transformers/torch are not installed; install the `nemotron` extra: uv sync --extra nemotron"
DEVICES = ("auto", "cpu", "mps", "cuda")
FRAME_MS = 80
DEFAULT_MAX_CHUNK_S = 360.0  # under the 400 s (5000-frame) positional-embedding limit
DEFAULT_SEARCH_S = 10.0
_LOCALES = {"en": "en-US", "es": "es-US", "fr": "fr-FR", "pt": "pt-BR", "de": "de-DE", "it": "it-IT", "nl": "nl-NL"}


def _import_transformers():
    try:
        import torch
        from transformers import AutoModelForRNNT, AutoProcessor
    except ImportError as exc:
        raise ProviderError(INSTALL_HINT) from exc
    return torch, AutoModelForRNNT, AutoProcessor


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class NemotronAsrProvider(Provider):
    key = "nemotron-asr"
    option_names = frozenset(
        {"device", "lookahead_frames", "language", "max_chunk_s", "search_s", "revision", "download_root"}
    )

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        options = config.options
        self.device = str(options.get("device", "auto"))
        self.lookahead_frames = options.get("lookahead_frames")  # None -> the checkpoint's default
        self.language = options.get("language")  # e.g. "en-US"; None -> derived from the dataset language
        self.max_chunk_s = options.get("max_chunk_s", DEFAULT_MAX_CHUNK_S)
        self.search_s = options.get("search_s", DEFAULT_SEARCH_S)
        self.revision = options.get("revision")
        self.download_root = options.get("download_root")
        self._model: Any | None = None
        self._processor: Any | None = None
        self._resolved_device: str | None = None

    def check(self) -> list[str]:
        problems = super().check()
        try:
            _import_transformers()
        except ProviderError as exc:
            problems.append(str(exc))
        if self.device not in DEVICES:
            problems.append(f"device must be one of {', '.join(DEVICES)}; got {self.device!r}")
        if self.lookahead_frames is not None and not (_is_int(self.lookahead_frames) and self.lookahead_frames >= 0):
            problems.append("lookahead_frames must be a non-negative integer (80 ms encoder frames)")
        if self.language is not None and not isinstance(self.language, str):
            problems.append("language must be a locale string such as en-US")
        for name in ("max_chunk_s", "search_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                problems.append(f"{name} must be a positive number of seconds")
        if isinstance(self.max_chunk_s, (int, float)) and self.max_chunk_s > 400:
            problems.append("max_chunk_s must be at most 400 (the model accepts 5000 encoder frames per request)")
        for name in ("revision", "download_root"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                problems.append(f"{name} must be a string")
        return problems

    # -- model loading -------------------------------------------------------------------------

    def _pick_device(self, torch) -> str:
        if self.device != "auto":
            return self.device
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load(self):
        if self._model is not None:
            return self._model, self._processor
        torch, AutoModelForRNNT, AutoProcessor = _import_transformers()
        kwargs: dict[str, Any] = {}
        if self.revision is not None:
            kwargs["revision"] = self.revision
        if self.download_root is not None:
            kwargs["cache_dir"] = self.download_root
        try:
            processor = AutoProcessor.from_pretrained(self.config.model, **kwargs)
            model = AutoModelForRNNT.from_pretrained(self.config.model, **kwargs)
        except Exception as exc:
            raise ProviderError(f"could not load {self.config.model!r} with transformers: {exc}") from exc
        if self.lookahead_frames is not None:
            supported = list(getattr(processor, "supported_num_lookahead_tokens", []) or [])
            if supported and self.lookahead_frames not in supported:
                raise ProviderError(
                    f"{self.config.model} supports lookahead_frames in {supported}; got {self.lookahead_frames}"
                )
            processor.set_num_lookahead_tokens(int(self.lookahead_frames))
        self._resolved_device = self._pick_device(torch)
        self._model = model.to(self._resolved_device).eval()
        self._processor = processor
        return self._model, self._processor

    # -- inference -----------------------------------------------------------------------------

    def _locale(self, language: str) -> str | None:
        if self.language is not None:
            return str(self.language)
        tag = language.strip()
        if "-" in tag:
            return tag
        return _LOCALES.get(tag.lower(), tag)

    @staticmethod
    def _accepts_language(processor) -> bool:
        # Only the multilingual (3.5) processor takes a language prompt; the English model does not.
        import inspect

        try:
            return "language" in inspect.signature(processor.__call__).parameters
        except (TypeError, ValueError):
            return False

    def _transcribe_chunk(self, torch, model, processor, samples, locale: str | None) -> str:
        import warnings

        kwargs: dict[str, Any] = {}
        if locale is not None and self._accepts_language(processor):
            kwargs["language"] = locale
        inputs = processor(samples, sampling_rate=16_000, return_tensors="pt", **kwargs).to(self._resolved_device)
        with warnings.catch_warnings(), torch.inference_mode():
            warnings.simplefilter("ignore")
            output = model.generate(**inputs, return_dict_in_generate=True)
        decoded = processor.decode(output.sequences, skip_special_tokens=True)
        text = decoded if isinstance(decoded, str) else decoded[0]
        return str(text or "").strip()

    def transcribe(self, audio: Path, *, language: str) -> Hypothesis:
        import soundfile as sf

        model, processor = self._load()
        torch, _, _ = _import_transformers()
        locale = self._locale(language)
        duration_s, _ = audio_info(audio)
        workdir = Path(tempfile.mkdtemp(prefix="ocular-evals-nemotron-"))
        try:
            chunks = split_wav16k(
                audio, max_chunk_s=float(self.max_chunk_s), search_s=float(self.search_s), workdir=workdir
            )
            segments: list[dict] = []
            texts: list[str] = []
            for index, (chunk_path, start_s) in enumerate(chunks):
                samples, rate = sf.read(str(chunk_path), dtype="float32")
                try:
                    text = self._transcribe_chunk(torch, model, processor, samples, locale)
                except Exception as exc:
                    raise ProviderError(f"{self.config.model} failed on {audio.name} chunk {index}: {exc}") from exc
                if text:
                    texts.append(text)
                segments.append({"start_s": start_s, "end_s": start_s + samples.size / float(rate), "text": text})
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        lookahead = getattr(processor, "default_num_lookahead_tokens", self.lookahead_frames)
        return Hypothesis(
            text=" ".join(texts),
            language=locale,
            duration_s=duration_s,
            segments=tuple(segments),
            raw={
                "model": self.config.model,
                "revision": self.revision,
                "device": self._resolved_device,
                "lookahead_frames": lookahead,
                "chunk_ms": (int(lookahead) + 1) * FRAME_MS if lookahead is not None else None,
                "language": locale,
                "max_chunk_s": float(self.max_chunk_s),
                "chunks": len(segments),
                "transformers": _versions(),
            },
        )

    def close(self) -> None:
        self._model = None
        self._processor = None


def _versions() -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        import torch
        import transformers

        out["transformers"] = str(transformers.__version__)
        out["torch"] = str(torch.__version__)
    except ImportError:
        pass
    return out


__all__ = ["NemotronAsrProvider"]
