"""Local inference for Cohere Transcribe (``CohereLabs/cohere-transcribe-03-2026``) via 🤗 Transformers.

A 2B-parameter Conformer encoder / Transformer decoder trained for offline transcription in 14
languages; weights are Apache-2.0 on Hugging Face behind a click-through, so a Hugging Face token
that has accepted the model's conditions must be available to ``huggingface_hub`` (``HF_TOKEN`` or
``hf auth login``). Installed by the ``nemotron`` extra (torch + transformers); nothing else is needed.

Long audio: the model's feature extractor splits anything over ``max_audio_clip_s`` (35 s) into
pieces at the quietest point in the last 5 s of each window. This adapter uses that same splitter
(``feature_extractor._split_audio_chunks_energy``) on the whole 16 kHz track so the cuts are the
model's own, transcribes the pieces in sub-batches of ``batch_size`` to bound memory on a laptop GPU,
and joins the non-empty texts with a space exactly as ``processor.decode(..., audio_chunk_index=)``
would. Per-piece texts and start times are kept as segments for the audit trail.

The model has no voice-activity gate and its card notes that it "is eager to transcribe, even
non-speech sounds"; the harness sends every piece, including silence, unless ``skip_silence_dbfs``
is set, in which case pieces whose peak level is below that threshold are skipped and recorded as
empty segments. The board run documents which setting it used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ocular_evals.config import ModelConfig
from ocular_evals.providers.audio import audio_info, load_mono16k
from ocular_evals.providers.base import Hypothesis, Provider, ProviderError, primary_language_tag

INSTALL_HINT = "transformers/torch are not installed; install the `nemotron` extra: uv sync --extra nemotron"
DEVICES = ("auto", "cpu", "mps", "cuda")
DTYPES = ("auto", "float32", "float16", "bfloat16")
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_NEW_TOKENS = 256
SAMPLE_RATE = 16_000


def _import_transformers():
    try:
        import torch
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration
    except ImportError as exc:
        raise ProviderError(INSTALL_HINT) from exc
    return torch, CohereAsrForConditionalGeneration, AutoProcessor


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _as_numpy(waveform):
    return waveform.numpy() if hasattr(waveform, "numpy") else waveform


def _peak_dbfs(samples) -> float:
    import math

    peak = float(abs(samples).max()) if samples.size else 0.0
    return 20.0 * math.log10(peak) if peak > 0 else -math.inf


class CohereTranscribeProvider(Provider):
    key = "cohere-transcribe"
    option_names = frozenset(
        {
            "device",
            "dtype",
            "batch_size",
            "max_new_tokens",
            "punctuation",
            "language",
            "skip_silence_dbfs",
            "revision",
            "download_root",
        }
    )

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        options = config.options
        self.device = str(options.get("device", "auto"))
        self.dtype = str(options.get("dtype", "auto"))
        self.batch_size = options.get("batch_size", DEFAULT_BATCH_SIZE)
        self.max_new_tokens = options.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
        self.punctuation = options.get("punctuation", True)
        self.language = options.get("language")  # e.g. "en"; None -> derived from the dataset language
        self.skip_silence_dbfs = options.get("skip_silence_dbfs")  # None -> transcribe every piece
        self.revision = options.get("revision")
        self.download_root = options.get("download_root")
        self._model: Any | None = None
        self._processor: Any | None = None
        self._resolved_device: str | None = None
        self._resolved_dtype: str | None = None

    def check(self) -> list[str]:
        problems = super().check()
        try:
            _import_transformers()
        except ProviderError as exc:
            problems.append(str(exc))
        if self.device not in DEVICES:
            problems.append(f"device must be one of {', '.join(DEVICES)}; got {self.device!r}")
        if self.dtype not in DTYPES:
            problems.append(f"dtype must be one of {', '.join(DTYPES)}; got {self.dtype!r}")
        for name in ("batch_size", "max_new_tokens"):
            value = getattr(self, name)
            if not (_is_int(value) and value > 0):
                problems.append(f"{name} must be a positive integer")
        if not isinstance(self.punctuation, bool):
            problems.append("punctuation must be true or false")
        if self.language is not None and not isinstance(self.language, str):
            problems.append("language must be a language code such as en")
        if self.skip_silence_dbfs is not None and (
            isinstance(self.skip_silence_dbfs, bool) or not isinstance(self.skip_silence_dbfs, (int, float))
        ):
            problems.append("skip_silence_dbfs must be a number of dBFS (for example -60)")
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

    def _pick_dtype(self, torch, device: str):
        if self.dtype != "auto":
            return getattr(torch, self.dtype)
        return torch.float32 if device == "cpu" else torch.bfloat16

    def _load(self):
        if self._model is not None:
            return self._model, self._processor
        torch, model_cls, AutoProcessor = _import_transformers()
        kwargs: dict[str, Any] = {}
        if self.revision is not None:
            kwargs["revision"] = self.revision
        if self.download_root is not None:
            kwargs["cache_dir"] = self.download_root
        device = self._pick_device(torch)
        dtype = self._pick_dtype(torch, device)
        try:
            processor = AutoProcessor.from_pretrained(self.config.model, **kwargs)
            model = model_cls.from_pretrained(self.config.model, dtype=dtype, **kwargs)
        except Exception as exc:
            raise ProviderError(f"could not load {self.config.model!r} with transformers: {exc}") from exc
        self._resolved_device = device
        self._resolved_dtype = str(dtype).replace("torch.", "")
        self._model = model.to(device).eval()
        self._processor = processor
        return self._model, self._processor

    # -- inference -----------------------------------------------------------------------------

    def _language(self, language: str) -> str:
        return str(self.language) if self.language is not None else primary_language_tag(language)

    def _pieces(self, torch, processor, samples) -> list[tuple[float, Any]]:
        """Cut the track with the model's own energy-based splitter; returns ``(start_s, waveform)``."""
        waveform = torch.from_numpy(samples)
        extractor = processor.feature_extractor
        splitter = getattr(extractor, "_split_audio_chunks_energy", None)
        chunks = splitter(waveform) if splitter is not None else [waveform]
        pieces: list[tuple[float, Any]] = []
        start = 0
        for chunk in chunks:
            pieces.append((start / SAMPLE_RATE, chunk))
            start += int(chunk.shape[0])
        return pieces

    def _transcribe_batch(self, torch, model, processor, waveforms: list, language: str) -> list[str]:
        import warnings

        inputs = processor(
            [_as_numpy(w) for w in waveforms],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            language=language,
            punctuation=bool(self.punctuation),
        )
        inputs.pop("audio_chunk_index", None)
        inputs = inputs.to(self._resolved_device)
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(model.dtype)
        with warnings.catch_warnings(), torch.inference_mode():
            warnings.simplefilter("ignore")
            output = model.generate(**inputs, max_new_tokens=int(self.max_new_tokens))
        texts = processor.tokenizer.batch_decode(output, skip_special_tokens=True)
        return [str(text or "").strip() for text in texts]

    def transcribe(self, audio: Path, *, language: str) -> Hypothesis:
        model, processor = self._load()
        torch, _, _ = _import_transformers()
        lang = self._language(language)
        duration_s, _ = audio_info(audio)
        samples = load_mono16k(audio)
        pieces = self._pieces(torch, processor, samples)

        segments: list[dict[str, Any]] = []
        todo: list[int] = []
        skipped = 0
        for index, (start_s, waveform) in enumerate(pieces):
            end_s = start_s + waveform.shape[0] / SAMPLE_RATE
            segments.append({"start_s": start_s, "end_s": end_s, "text": ""})
            if self.skip_silence_dbfs is not None and _peak_dbfs(_as_numpy(waveform)) < float(self.skip_silence_dbfs):
                skipped += 1
                continue
            todo.append(index)

        size = int(self.batch_size)
        for offset in range(0, len(todo), size):
            batch = todo[offset : offset + size]
            try:
                texts = self._transcribe_batch(torch, model, processor, [pieces[i][1] for i in batch], lang)
            except Exception as exc:
                raise ProviderError(f"{self.config.model} failed on {audio.name} pieces {batch}: {exc}") from exc
            for index, text in zip(batch, texts):
                segments[index]["text"] = text

        texts = [segment["text"] for segment in segments if segment["text"]]
        return Hypothesis(
            text=" ".join(texts),
            language=lang,
            duration_s=duration_s,
            segments=tuple(segments),
            raw={
                "model": self.config.model,
                "revision": self.revision,
                "device": self._resolved_device,
                "dtype": self._resolved_dtype,
                "language": lang,
                "punctuation": bool(self.punctuation),
                "max_new_tokens": int(self.max_new_tokens),
                "batch_size": size,
                "pieces": len(pieces),
                "skip_silence_dbfs": self.skip_silence_dbfs,
                "pieces_skipped_as_silence": skipped,
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


__all__ = ["CohereTranscribeProvider"]
