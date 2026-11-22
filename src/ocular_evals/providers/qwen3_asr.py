"""Offline short-window transcription with native Transformers Qwen3-ASR checkpoints."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from ocular_evals.config import ModelConfig
from ocular_evals.providers.audio import load_mono16k, split_wav16k
from ocular_evals.providers.base import Hypothesis, Provider, ProviderError, primary_language_tag


def _runtime():
    try:
        import torch
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration
    except ImportError as exc:
        hint = "Install PyTorch and Transformers with native Qwen3-ASR support (tested with 5.24.0)"
        raise ProviderError(hint) from exc
    return torch, AutoProcessor, Qwen3ASRForConditionalGeneration


class Qwen3AsrProvider(Provider):
    key = "qwen3-asr"
    option_names = frozenset({"revision", "device"})

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._processor = None
        self._model = None

    def check(self) -> list[str]:
        problems = super().check()
        if self.config.options.get("device", "auto") not in ("auto", "cpu", "cuda"):
            problems.append("device must be auto, cpu or cuda")
        revision = self.config.options.get("revision")
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            problems.append("revision must be a nonempty Hub revision")
        try:
            _runtime()
        except ProviderError as exc:
            problems.append(str(exc))
        return problems

    def transcribe(self, audio: Path, *, language: str) -> Hypothesis:
        torch, processor_class, model_class = _runtime()
        if self._model is None:
            device = self.config.options.get("device", "auto")
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
            revision = self.config.options.get("revision")
            self._processor = processor_class.from_pretrained(self.config.model, revision=revision)
            model = model_class.from_pretrained(self.config.model, revision=revision, dtype=dtype)
            self._model = model.to(device).eval()

        lang = primary_language_tag(language)
        segments = []
        with TemporaryDirectory(prefix="ocular-qwen-asr-") as tmp, torch.inference_mode():
            for path, start in split_wav16k(audio, max_chunk_s=30.0, search_s=2.0, workdir=Path(tmp)):
                samples = load_mono16k(path)
                if not samples.size:
                    continue
                request = self._processor.apply_transcription_request(audio=samples, language=lang, return_tensors="pt")
                request = request.to(self._model.device, self._model.dtype)
                output = self._model.generate(**request)
                generated = output[:, request["input_ids"].shape[1] :]
                text = self._processor.decode(generated, return_format="transcription_only")[0].strip()
                segments.append({"start_s": start, "end_s": start + samples.size / 16_000, "text": text})
        return Hypothesis(
            text=" ".join(seg["text"] for seg in segments if seg["text"]),
            duration_s=segments[-1]["end_s"] if segments else 0.0,
            language=lang,
            segments=tuple(segments),
        )

    def close(self) -> None:
        self._processor = None
        self._model = None
