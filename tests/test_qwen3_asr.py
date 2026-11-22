from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import soundfile as sf

from ocular_evals.config import load_model_config
from ocular_evals.providers import create_provider, qwen3_asr


def test_native_profile_passes_revision_to_both_loaders_and_reuses_model(tmp_path, monkeypatch):
    profile = load_model_config("oruk-fourier-2b")
    provider = create_provider(profile)
    model = Mock()
    model.to.return_value.eval.return_value = model
    model.generate.return_value = np.array([[102, 103, 104]])
    model_class = Mock()
    model_class.from_pretrained.return_value = model

    request = Mock()
    request.to.return_value = {"input_ids": np.array([[102, 103]])}
    processor = Mock()
    processor.apply_transcription_request.return_value = request
    processor.decode.return_value = [" speech "]
    processor_class = Mock()
    processor_class.from_pretrained.return_value = processor
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False), float32="float32", inference_mode=nullcontext
    )
    monkeypatch.setattr(qwen3_asr, "_runtime", lambda: (torch, processor_class, model_class))

    audio = tmp_path / "clip.wav"
    sf.write(audio, np.ones(32_000, dtype=np.float32) / 4, 16_000)
    assert provider.check() == []
    for _ in range(2):
        hypothesis = provider.transcribe(audio, language="en-US")
        assert hypothesis.text == "speech" and hypothesis.duration_s == 2 and hypothesis.language == "en"
        assert hypothesis.segments == ({"text": "speech", "start_s": 0, "end_s": 2},)
    revision = profile.options["revision"]
    processor_class.from_pretrained.assert_called_once_with(profile.model, revision=revision)
    model_class.from_pretrained.assert_called_once_with(profile.model, revision=revision, dtype="float32")
    assert processor.apply_transcription_request.call_args.kwargs["language"] == "en"
    assert processor.decode.call_args.args[0].tolist() == [[104]]
    assert processor.decode.call_args.kwargs == {"return_format": "transcription_only"}
    provider.close()
