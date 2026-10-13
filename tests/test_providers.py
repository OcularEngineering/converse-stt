from __future__ import annotations

import builtins
import sys
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import soundfile as sf

from ocular_evals.config import ModelConfig
from ocular_evals.providers import PROVIDERS, Hypothesis, ProviderError, create_provider
from ocular_evals.providers.audio import audio_info, content_type_for, prepare_upload, split_wav16k
from ocular_evals.providers.faster_whisper import FasterWhisperProvider
from ocular_evals.providers.http import error_from_response, send_with_retry
from ocular_evals.providers.meta import MetaProvider
from ocular_evals.providers.nemotron_asr import NemotronAsrProvider
from ocular_evals.providers.omi import OmiProvider
from ocular_evals.providers.openai_compatible import OpenAICompatibleProvider
from ocular_evals.providers.openai_realtime import OpenAIRealtimeProvider, _load_pcm24k
from ocular_evals.providers.reson8 import Reson8Provider
from ocular_evals.providers.xai import XaiProvider

# --- helpers -------------------------------------------------------------------


def _response(status: int, url: str, *, json: Any = None, text: str | None = None, headers=None) -> httpx.Response:
    request = httpx.Request("POST", url)
    if text is not None:
        return httpx.Response(status, text=text, headers=headers, request=request)
    return httpx.Response(status, json=json, headers=headers, request=request)


class FakeClient:
    """Records every call and answers from a ``route(method, url, kwargs)`` callback."""

    def __init__(self, route) -> None:
        self.route = route
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def _call(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        files = kwargs.get("files")
        if files:
            # Read the upload now; the provider closes the handle when the call returns.
            name, handle, content_type = files["file"]
            kwargs = {**kwargs, "files": {"file": (name, handle.read(), content_type)}}
        self.calls.append((method, url, kwargs))
        return self.route(method, url, kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self._call("POST", url, **kwargs)

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self._call("GET", url, **kwargs)

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        return self._call(method.upper(), url, **kwargs)

    def close(self) -> None:
        return None


def _config(provider: str, model: str, **options: Any) -> ModelConfig:
    env = {
        "omi": ("OMI_API_KEY",),
        "openai-compatible": ("OPENAI_API_KEY",),
        "openai-realtime": ("OPENAI_API_KEY",),
        "xai": ("XAI_API_KEY",),
        "reson8": ("RESON8_API_KEY",),
        "meta": ("META_API_KEY",),
    }.get(provider, ())
    return ModelConfig(
        name=model, display_name=model, vendor=provider, provider=provider, model=model, env=env, options=options
    )


@pytest.fixture
def flac(tmp_path: Path) -> Path:
    path = tmp_path / "speaker.flac"
    sf.write(str(path), np.zeros((1600, 1), dtype=np.float32), 16000, format="FLAC")
    return path


# --- registry ------------------------------------------------------------------


def test_registry_exposes_every_provider():
    assert set(PROVIDERS) == {
        "openai-compatible",
        "openai-realtime",
        "xai",
        "meta",
        "reson8",
        "omi",
        "faster-whisper",
        "nemotron-asr",
    }
    assert isinstance(create_provider(_config("omi", "omi-general-1")), OmiProvider)
    assert isinstance(create_provider(_config("xai", "grok-voice-transcribe-2.0")), XaiProvider)
    assert isinstance(create_provider(_config("meta", "muse-voice-transcribe-1.0")), MetaProvider)


# --- send_with_retry -----------------------------------------------------------


def test_send_with_retry_honors_retry_after_then_returns_200():
    url = "https://api.example/thing"
    responses = [
        _response(429, url, json={}, headers={"Retry-After": "7"}),
        _response(200, url, json={"ok": True}),
    ]
    sleeps: list[float] = []

    result = send_with_retry(lambda: responses.pop(0), sleep=sleeps.append, base_delay_s=0.5, max_delay_s=30.0)

    assert result.status_code == 200
    assert result.json() == {"ok": True}
    assert len(sleeps) == 1 and 7.0 <= sleeps[0] <= 30.0
    assert responses == []


def test_send_with_retry_raises_retryable_after_exhausting_attempts():
    url = "https://api.example/thing"
    sleeps: list[float] = []
    calls = 0

    def send() -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response(503, url, json={"error": "busy"})

    with pytest.raises(ProviderError) as info:
        send_with_retry(send, attempts=3, sleep=sleeps.append, base_delay_s=1.0, max_delay_s=4.0)

    assert info.value.retryable is True
    assert info.value.status == 503
    assert calls == 3
    assert len(sleeps) == 2
    assert all(delay <= 4.0 for delay in sleeps)


def test_send_with_retry_retries_transport_errors():
    url = "https://api.example/thing"
    attempts = iter([httpx.ConnectError("boom"), httpx.ReadTimeout("slow"), _response(200, url, json={})])
    sleeps: list[float] = []

    def send() -> httpx.Response:
        item = next(attempts)
        if isinstance(item, Exception):
            raise item
        return item

    assert send_with_retry(send, sleep=sleeps.append).status_code == 200
    assert len(sleeps) == 2


def test_send_with_retry_passes_400_through_without_retry():
    url = "https://api.example/thing"
    sleeps: list[float] = []
    calls = 0

    def send() -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response(400, url, json={"error": "bad"})

    response = send_with_retry(send, sleep=sleeps.append)

    assert response.status_code == 400
    assert calls == 1
    assert sleeps == []


def test_send_with_retry_stops_immediately_on_exhausted_quota_429():
    url = "https://api.openai.com/v1/audio/transcriptions"
    sleeps: list[float] = []
    calls = 0
    body = {
        "error": {
            "message": "You have no credits remaining. Add credits to continue using the API.",
            "type": "insufficient_quota",
            "code": "credit_balance_exhausted",
        }
    }

    def send() -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response(429, url, json=body)

    response = send_with_retry(send, sleep=sleeps.append)
    assert response.status_code == 429 and calls == 1 and sleeps == []

    error = error_from_response(response, context="transcription failed")
    assert error.retryable is False and error.status == 429
    assert "insufficient_quota" in str(error) and "not retrying" in str(error)


def test_send_with_retry_final_error_carries_the_body():
    url = "https://api.example/thing"

    def send() -> httpx.Response:
        return _response(503, url, json={"error": "upstream warming up"})

    with pytest.raises(ProviderError, match="HTTP 503.*upstream warming up"):
        send_with_retry(send, attempts=2, sleep=lambda _: None)


def test_error_from_response_redacts_and_truncates(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-abcdefghijklmnopqrstuvwxyz")
    body = "leaked sk-abcdefghijklmnopqrstuvwxyz " + "x" * 1000
    error = error_from_response(_response(401, "https://api.example/x", text=body), context="ctx")

    assert error.status == 401
    assert error.retryable is False
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in str(error)
    assert "[redacted]" in str(error)
    assert "HTTP 401" in str(error)
    assert len(str(error)) < 400


# --- OpenAI-compatible ---------------------------------------------------------


def test_openai_compatible_posts_multipart_and_parses_json(monkeypatch: pytest.MonkeyPatch, flac: Path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-testtesttesttesttesttest")

    def route(method: str, url: str, kwargs: dict[str, Any]) -> httpx.Response:
        assert method == "POST"
        return _response(
            200,
            url,
            json={
                "text": " hello world ",
                "language": "en",
                "duration": 1.5,
                "segments": [{"start": 0.0, "end": 0.7, "text": "hello"}, {"start": 0.8, "end": 1.5, "text": "world"}],
            },
            headers={"x-request-id": "req_123"},
        )

    client = FakeClient(route)
    provider = OpenAICompatibleProvider(_config("openai-compatible", "gpt-transcribe", temperature=0.2), client=client)

    hypothesis = provider.transcribe(flac, language="en-US")

    assert isinstance(hypothesis, Hypothesis)
    assert hypothesis.text == "hello world"
    assert hypothesis.language == "en"
    assert hypothesis.duration_s == 1.5
    assert hypothesis.segments == (
        {"start_s": 0.0, "end_s": 0.7, "text": "hello"},
        {"start_s": 0.8, "end_s": 1.5, "text": "world"},
    )
    method, url, kwargs = client.calls[0]
    assert url == "https://api.openai.com/v1/audio/transcriptions"
    assert kwargs["headers"]["Authorization"] == "Bearer sk-testtesttesttesttesttest"
    assert kwargs["data"]["model"] == "gpt-transcribe"
    assert kwargs["data"]["language"] == "en"
    assert kwargs["data"]["response_format"] == "json"
    assert kwargs["data"]["temperature"] == "0.2"
    name, payload, content_type = kwargs["files"]["file"]
    assert name == "speaker.flac"
    assert content_type == "audio/flac"
    assert payload == flac.read_bytes()
    assert hypothesis.raw["endpoint"] == url
    assert hypothesis.raw["parameters"]["model"] == "gpt-transcribe"
    assert hypothesis.raw["upload"] == {"format": "flac", "content_type": "audio/flac", "bytes": flac.stat().st_size}
    assert hypothesis.raw["request_id"] == "req_123"
    assert "sk-test" not in repr(hypothesis.raw)


def test_openai_compatible_text_format_and_custom_base_url(monkeypatch: pytest.MonkeyPatch, flac: Path):
    monkeypatch.setenv("MY_KEY", "token-value-1234567890")
    client = FakeClient(lambda method, url, kwargs: _response(200, url, text="plain words\n"))
    provider = OpenAICompatibleProvider(
        _config(
            "openai-compatible",
            "whisper-1",
            base_url="http://localhost:8000/v1/",
            api_key_env="MY_KEY",
            response_format="text",
            send_language=False,
        ),
        client=client,
    )

    hypothesis = provider.transcribe(flac, language="en")

    assert hypothesis.text == "plain words"
    _, url, kwargs = client.calls[0]
    assert url == "http://localhost:8000/v1/audio/transcriptions"
    assert "language" not in kwargs["data"]
    assert kwargs["headers"]["Authorization"] == "Bearer token-value-1234567890"


def test_openai_compatible_sends_chunking_strategy_when_set(monkeypatch: pytest.MonkeyPatch, flac: Path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-testtesttesttesttesttest")
    client = FakeClient(lambda method, url, kwargs: _response(200, url, json={"text": "ok"}))
    plain = OpenAICompatibleProvider(_config("openai-compatible", "gpt-transcribe"), client=client)
    assert plain.check() == []
    plain.transcribe(flac, language="en")
    assert "chunking_strategy" not in client.calls[0][2]["data"]

    chunked = OpenAICompatibleProvider(
        _config("openai-compatible", "gpt-4o-transcribe", chunking_strategy="auto"), client=client
    )
    assert chunked.check() == []
    hypothesis = chunked.transcribe(flac, language="en")
    assert client.calls[1][2]["data"]["chunking_strategy"] == "auto"
    assert hypothesis.raw["parameters"]["chunking_strategy"] == "auto"
    bad = OpenAICompatibleProvider(_config("openai-compatible", "gpt-4o-transcribe", chunking_strategy=3))
    assert any("chunking_strategy" in item for item in bad.check())

    stamped = OpenAICompatibleProvider(
        _config("openai-compatible", "voxtral-mini-2602", timestamp_granularities="segment"), client=client
    )
    assert stamped.check() == []
    stamped.transcribe(flac, language="en")
    assert client.calls[2][2]["data"]["timestamp_granularities"] == "segment"
    listed = OpenAICompatibleProvider(
        _config("openai-compatible", "whisper-1", timestamp_granularities=["segment", "word"]), client=client
    )
    listed.transcribe(flac, language="en")
    assert client.calls[3][2]["data"]["timestamp_granularities"] == ["segment", "word"]
    assert OpenAICompatibleProvider(_config("openai-compatible", "x", timestamp_granularities=1)).check()
    assert stamped.config.digest() != plain.config.digest()
    # The option changes what comes back, so it must change the digest that keys resumable runs.
    assert plain.config.digest() != chunked.config.digest()


def test_openai_compatible_chunk_s_splits_and_joins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-testtesttesttesttesttest")
    src = _long_flac(tmp_path, 20.0, quiet_at=(9.5, 10.5))
    answers = iter(
        [
            {
                "text": "first part",
                "duration": 10.0,
                "language": "en",
                "segments": [{"start": 1.0, "end": 9.0, "text": "first part"}],
            },
            {"text": "second part", "duration": 10.0, "segments": [{"start": 0.5, "end": 9.5, "text": "second part"}]},
        ]
    )

    def route(method: str, url: str, kwargs: dict[str, Any]) -> httpx.Response:
        name, payload, content_type = kwargs["files"]["file"]
        assert name.endswith(".wav") and content_type == "audio/wav" and payload.startswith(b"RIFF")
        assert kwargs["data"]["model"] == "google/chirp-3"
        return _response(200, url, json=next(answers))

    client = FakeClient(route)
    provider = OpenAICompatibleProvider(
        _config("openai-compatible", "google/chirp-3", chunk_s=15, search_s=4), client=client
    )
    assert provider.check() == []

    hypothesis = provider.transcribe(src, language="en")

    assert hypothesis.text == "first part second part"
    assert hypothesis.duration_s == 20.0
    assert hypothesis.language == "en"
    assert len(client.calls) == 2
    assert hypothesis.segments[0] == {"start_s": 1.0, "end_s": 9.0, "text": "first part"}
    assert 10.0 <= hypothesis.segments[1]["start_s"] <= 11.0  # offset by the second chunk's start
    assert hypothesis.raw["upload"]["chunk_s"] == 15 and len(hypothesis.raw["upload"]["chunks"]) == 2
    assert "chunk 2/2" in capsys.readouterr().err
    assert OpenAICompatibleProvider(_config("openai-compatible", "x", chunk_s=0)).check()
    assert OpenAICompatibleProvider(_config("openai-compatible", "x", search_s=-1)).check()
    provider.close()


def test_openai_compatible_202_is_non_retryable_and_mentions_omi(monkeypatch: pytest.MonkeyPatch, flac: Path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-testtesttesttesttesttest")
    client = FakeClient(lambda method, url, kwargs: _response(202, url, json={"job_id": "j1"}))
    provider = OpenAICompatibleProvider(_config("openai-compatible", "gpt-transcribe"), client=client)

    with pytest.raises(ProviderError) as info:
        provider.transcribe(flac, language="en")

    assert info.value.retryable is False
    assert info.value.status == 202
    assert "omi" in str(info.value)


def test_openai_compatible_4xx_becomes_provider_error(monkeypatch: pytest.MonkeyPatch, flac: Path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-testtesttesttesttesttest")
    client = FakeClient(lambda method, url, kwargs: _response(401, url, json={"error": "nope"}))
    provider = OpenAICompatibleProvider(_config("openai-compatible", "gpt-transcribe"), client=client)

    with pytest.raises(ProviderError) as info:
        provider.transcribe(flac, language="en")

    assert info.value.status == 401
    assert info.value.retryable is False


def test_openai_compatible_check_reports_missing_env_and_bad_options(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = OpenAICompatibleProvider(_config("openai-compatible", "gpt-transcribe"))
    problems = provider.check()
    assert any("OPENAI_API_KEY" in item for item in problems)

    bad = OpenAICompatibleProvider(
        _config("openai-compatible", "gpt-transcribe", response_format="srt", upload="aiff", bogus=1)
    )
    problems = bad.check()
    assert any("response_format" in item for item in problems)
    assert any("upload" in item for item in problems)
    assert any("unknown option" in item and "bogus" in item for item in problems)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-testtesttesttesttesttest")
    monkeypatch.setattr("ocular_evals.providers.openai_compatible.ffmpeg_available", lambda: False)
    assert OpenAICompatibleProvider(_config("openai-compatible", "gpt-transcribe")).check() == []
    mp3 = OpenAICompatibleProvider(_config("openai-compatible", "gpt-transcribe", upload="mp3"))
    assert any("ffmpeg" in item for item in mp3.check())


# --- xAI -----------------------------------------------------------------------

XAI_KEY = "xai-secretsecretsecret"


def test_xai_posts_to_stt_with_flags_and_parses_words(monkeypatch: pytest.MonkeyPatch, flac: Path):
    monkeypatch.setenv("XAI_API_KEY", XAI_KEY)

    def route(method: str, url: str, kwargs: dict[str, Any]) -> httpx.Response:
        assert method == "POST"
        return _response(
            200,
            url,
            json={
                "text": " Uh, hello world. ",
                "language": "en",
                "duration": 1.5,
                "words": [{"text": "Uh,", "start": 0.0, "end": 0.3}, {"text": "hello", "start": 0.4, "end": 0.9}],
            },
            headers={"x-request-id": "req_x"},
        )

    client = FakeClient(route)
    provider = XaiProvider(_config("xai", "grok-voice-transcribe-2.0"), client=client)
    assert provider.check() == []

    hypothesis = provider.transcribe(flac, language="en-US")

    assert hypothesis.text == "Uh, hello world."
    assert hypothesis.language == "en"
    assert hypothesis.duration_s == 1.5
    assert hypothesis.segments[0] == {"start_s": 0.0, "end_s": 0.3, "text": "Uh,"}
    _, url, kwargs = client.calls[0]
    assert url == "https://api.x.ai/v1/stt"
    assert kwargs["headers"]["Authorization"] == f"Bearer {XAI_KEY}"
    assert kwargs["data"] == {
        "model": "grok-voice-transcribe-2.0",
        "language": "en",
        "format": "true",
        "filler_words": "true",
    }
    name, payload, content_type = kwargs["files"]["file"]
    assert name == "speaker.flac" and content_type == "audio/flac" and payload == flac.read_bytes()
    assert hypothesis.raw["request_id"] == "req_x"
    assert hypothesis.raw["parameters"]["filler_words"] is True
    assert XAI_KEY not in repr(hypothesis.raw)


def test_xai_options_and_errors(monkeypatch: pytest.MonkeyPatch, flac: Path):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    problems = XaiProvider(_config("xai", "grok-voice-transcribe-2.0", format="yes", nope=1)).check()
    assert any("XAI_API_KEY" in item for item in problems)
    assert any("format" in item for item in problems)
    assert any("nope" in item for item in problems)

    monkeypatch.setenv("XAI_API_KEY", XAI_KEY)
    client = FakeClient(lambda method, url, kwargs: _response(401, url, json={"error": "nope"}))
    provider = XaiProvider(
        _config("xai", "grok-voice-transcribe-2.0", filler_words=False, send_language=False), client=client
    )
    with pytest.raises(ProviderError) as info:
        provider.transcribe(flac, language="en")
    assert info.value.status == 401 and info.value.retryable is False
    data = client.calls[0][2]["data"]
    assert "language" not in data and data["filler_words"] == "false"


# --- Reson8 --------------------------------------------------------------------

RESON8_KEY = "r8-secretsecretsecret"


def test_reson8_posts_raw_body_with_query_parameters(monkeypatch: pytest.MonkeyPatch, flac: Path):
    monkeypatch.setenv("RESON8_API_KEY", RESON8_KEY)

    def route(method: str, url: str, kwargs: dict[str, Any]) -> httpx.Response:
        assert method == "POST"
        return _response(
            200,
            url,
            json={
                "text": " uh hello world ",
                "language": "en",
                "start_ms": 0,
                "duration_ms": 1500,
                "words": [
                    {"text": "uh", "start_ms": 0, "duration_ms": 300},
                    {"text": "hello", "start_ms": 400, "duration_ms": 500},
                ],
            },
            headers={"x-request-id": "req_r8"},
        )

    client = FakeClient(route)
    provider = Reson8Provider(_config("reson8", "reson8"), client=client)
    assert provider.check() == []

    hypothesis = provider.transcribe(flac, language="en-US")

    assert hypothesis.text == "uh hello world"
    assert hypothesis.language == "en"
    assert hypothesis.duration_s == 1.5
    assert hypothesis.segments[1] == {"start_s": 0.4, "end_s": 0.9, "text": "hello"}
    _, url, kwargs = client.calls[0]
    assert url == "https://api.reson8.dev/v1/speech-to-text/prerecorded"
    assert kwargs["headers"] == {"Authorization": f"ApiKey {RESON8_KEY}", "Content-Type": "application/octet-stream"}
    assert kwargs["params"] == {
        "encoding": "auto",
        "filler_mode": "verbatim",
        "language": "en",
        "include_language": "true",
        "include_words": "true",
        "include_timestamps": "true",
    }
    assert kwargs["content"] == flac.read_bytes()
    assert hypothesis.raw["request_id"] == "req_r8"
    assert hypothesis.raw["response"]["text"] == " uh hello world "
    assert RESON8_KEY not in repr(hypothesis.raw)


def test_reson8_options_and_errors(monkeypatch: pytest.MonkeyPatch, flac: Path):
    monkeypatch.delenv("RESON8_API_KEY", raising=False)
    problems = Reson8Provider(_config("reson8", "reson8", filler_mode="loud", include_words="yes", nope=1)).check()
    assert any("RESON8_API_KEY" in item for item in problems)
    assert any("filler_mode" in item for item in problems)
    assert any("include_words" in item for item in problems)
    assert any("nope" in item for item in problems)

    monkeypatch.setenv("RESON8_API_KEY", RESON8_KEY)
    client = FakeClient(lambda method, url, kwargs: _response(402, url, json={"error": "credit limit"}))
    provider = Reson8Provider(
        _config("reson8", "reson8", filler_mode="clean", send_language=False, include_words=False), client=client
    )
    with pytest.raises(ProviderError) as info:
        provider.transcribe(flac, language="en")
    assert info.value.status == 402 and info.value.retryable is False
    params = client.calls[0][2]["params"]
    assert "language" not in params and "include_words" not in params and params["filler_mode"] == "clean"


# --- Meta ----------------------------------------------------------------------

META_KEY = "LLM_metasecretsecret"


def _long_flac(tmp_path: Path, seconds: float, *, quiet_at: tuple[float, float] | None = None) -> Path:
    rate = 16000
    t = np.arange(int(rate * seconds)) / rate
    signal = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    if quiet_at is not None:
        start, end = quiet_at
        signal[int(start * rate) : int(end * rate)] = 0.0
    path = tmp_path / "long.flac"
    sf.write(str(path), signal, rate, format="FLAC")
    return path


def test_split_wav16k_cuts_in_the_quiet_window(tmp_path: Path):
    src = _long_flac(tmp_path, 20.0, quiet_at=(12.0, 13.0))

    chunks = split_wav16k(src, max_chunk_s=15.0, search_s=4.0, frame_s=0.5, workdir=tmp_path / "work")

    assert len(chunks) == 2
    (first, start0), (second, start1) = chunks
    assert start0 == 0.0
    assert 12.0 <= start1 <= 13.0  # moved from the even split at 10 s into the silence
    assert sf.info(str(first)).duration + sf.info(str(second)).duration == pytest.approx(20.0, abs=1e-3)
    assert all(sf.info(str(path)).duration <= 15.0 for path, _ in chunks)
    assert sf.info(str(first)).samplerate == 16000 and sf.info(str(first)).subtype == "PCM_16"


def test_split_wav16k_single_chunk_when_it_fits(tmp_path: Path, flac: Path):
    chunks = split_wav16k(flac, max_chunk_s=600.0, workdir=tmp_path / "work")
    assert len(chunks) == 1 and chunks[0][1] == 0.0


def test_split_wav16k_never_exceeds_the_limit(tmp_path: Path):
    src = _long_flac(tmp_path, 30.0, quiet_at=(0.5, 1.0))  # the only silence is far from any even split
    chunks = split_wav16k(src, max_chunk_s=10.0, search_s=30.0, workdir=tmp_path / "work")
    assert len(chunks) == 3
    assert all(sf.info(str(path)).duration <= 10.0 + 1e-6 for path, _ in chunks)


def test_meta_chunks_long_audio_and_joins_transcripts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys):
    monkeypatch.setenv("META_API_KEY", META_KEY)
    src = _long_flac(tmp_path, 20.0, quiet_at=(9.5, 10.5))
    answers = iter(
        [
            {
                "sessionId": "s1",
                "transcript": "first half.",
                "audioDurationMs": 10000,
                "turns": [{"turnId": 0, "startMs": 0, "endMs": 9000, "transcript": "first half."}],
            },
            {
                "sessionId": "s2",
                "transcript": "second half",
                "audioDurationMs": 10000,
                "turns": [{"turnId": 0, "startMs": 500, "endMs": 9500, "transcript": "second half"}],
            },
        ]
    )

    def route(method: str, url: str, kwargs: dict[str, Any]) -> httpx.Response:
        assert method == "POST" and url == "https://api.meta.ai/v1/asr/transcribe"
        assert kwargs["headers"]["Authorization"] == f"Bearer {META_KEY}"
        assert "data" not in kwargs
        request_part = kwargs["files"]["request"]
        assert request_part[0] is None and request_part[2] == "application/json"
        import json

        assert json.loads(request_part[1]) == {
            "model": "muse-voice-transcribe-1.0",
            "audioEncoding": "WAV",
            "mode": "ENDPOINTING",
        }
        name, payload, content_type = kwargs["files"]["audio"]
        assert name.endswith(".wav") and content_type == "audio/wav" and payload.startswith(b"RIFF")
        return _response(200, url, json=next(answers))

    class ChunkAwareClient(FakeClient):
        def _call(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
            files = kwargs["files"]
            name, handle, content_type = files["audio"]
            kwargs = {**kwargs, "files": {**files, "audio": (name, handle.read(), content_type)}}
            self.calls.append((method, url, kwargs))
            return self.route(method, url, kwargs)

    client = ChunkAwareClient(route)
    provider = MetaProvider(_config("meta", "muse-voice-transcribe-1.0", chunk_s=15, search_s=4), client=client)
    assert provider.check() == []

    hypothesis = provider.transcribe(src, language="en-US")

    assert hypothesis.text == "first half. second half"
    assert hypothesis.language == "en"
    assert hypothesis.duration_s == 20.0
    assert len(client.calls) == 2
    assert len(hypothesis.segments) == 2
    assert hypothesis.segments[0] == {"start_s": 0.0, "end_s": 9.0, "text": "first half."}
    second = hypothesis.segments[1]
    assert 9.5 + 0.5 <= second["start_s"] <= 10.5 + 0.5  # offset by the chunk start
    assert hypothesis.raw["upload"]["format"] == "wav16k"
    assert [part["session_id"] for part in hypothesis.raw["upload"]["chunks"]] == ["s1", "s2"]
    assert META_KEY not in repr(hypothesis.raw)
    assert "chunk 2/2" in capsys.readouterr().err
    provider.close()


def test_meta_check_rejects_bad_mode_and_oversized_chunks(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("META_API_KEY", raising=False)
    problems = MetaProvider(_config("meta", "muse-voice-transcribe-1.0", mode="loud", chunk_s=900, extra=1)).check()
    assert any("META_API_KEY" in item for item in problems)
    assert any("mode" in item for item in problems)
    assert any("chunk_s" in item for item in problems)
    assert any("extra" in item for item in problems)
    monkeypatch.setenv("META_API_KEY", META_KEY)
    assert MetaProvider(_config("meta", "muse-voice-transcribe-1.0")).check() == []


# --- Omi -----------------------------------------------------------------------

OMI_KEY = "omi_sk_secretsecret"


def _omi_router(poll_states: list[dict[str, Any]], *, result_payload: dict[str, Any]):
    states = list(poll_states)

    def route(method: str, url: str, kwargs: dict[str, Any]) -> httpx.Response:
        if method == "POST" and url == "https://api.omi.health/v1/jobs":
            body = kwargs["json"]
            assert body["model"] == "omi-general-1"
            assert body["language"] == "en"
            assert body["response_format"] == "json"
            assert body["content_type"] == "audio/flac"
            assert body["filename"] == "speaker.flac"
            assert body["content_length_bytes"] > 0
            assert kwargs["headers"]["Authorization"] == f"Bearer {OMI_KEY}"
            return _response(
                201,
                url,
                json={
                    "job_id": "job_42",
                    "status": "awaiting_upload",
                    "upload": {"url": "https://storage.example/bucket", "fields": {"key": "obj/1", "policy": "p"}},
                    "complete_url": "/v1/jobs/job_42/complete",
                    "poll_url": "https://api.omi.health/v1/jobs/job_42",
                },
            )
        if method == "POST" and url == "https://storage.example/bucket":
            assert "Authorization" not in (kwargs.get("headers") or {})
            assert kwargs["data"] == {"key": "obj/1", "policy": "p"}
            name, payload, content_type = kwargs["files"]["file"]
            assert name == "speaker.flac" and content_type == "audio/flac" and payload.startswith(b"fLaC")
            return _response(204, url, text="")
        if method == "POST" and url == "https://api.omi.health/v1/jobs/job_42/complete":
            assert kwargs["headers"]["Authorization"] == f"Bearer {OMI_KEY}"
            return _response(200, url, json={"status": "queued"})
        if method == "GET" and url == "https://api.omi.health/v1/jobs/job_42":
            assert kwargs["headers"]["Authorization"] == f"Bearer {OMI_KEY}"
            assert kwargs["params"] == {"wait": 20, "include_result": "true"}
            return _response(200, url, json=states.pop(0))
        if method == "GET" and url == "https://results.example/job_42.json":
            assert kwargs["headers"] == {}
            return _response(200, url, json=result_payload)
        raise AssertionError(f"unexpected call {method} {url}")

    return route


def test_omi_job_flow_with_external_download(monkeypatch: pytest.MonkeyPatch, flac: Path, capsys):
    monkeypatch.setenv("OMI_API_KEY", OMI_KEY)
    sleeps: list[float] = []
    route = _omi_router(
        [
            {"status": "running"},
            {"status": "running"},
            {"status": "succeeded", "result": {"download_url": "https://results.example/job_42.json"}},
        ],
        result_payload={"text": "hello there", "language": "en", "duration": 2.0},
    )
    client = FakeClient(route)
    provider = OmiProvider(_config("omi", "omi-general-1"), client=client, sleep=sleeps.append)

    hypothesis = provider.transcribe(flac, language="en-GB")

    assert hypothesis.text == "hello there"
    assert hypothesis.language == "en"
    assert hypothesis.duration_s == 2.0
    assert hypothesis.raw["endpoint"] == "https://api.omi.health/v1/jobs"
    assert hypothesis.raw["job_id"] == "job_42"
    assert hypothesis.raw["parameters"] == {"model": "omi-general-1", "language": "en", "response_format": "json"}
    assert hypothesis.raw["upload"]["content_type"] == "audio/flac"
    assert OMI_KEY not in repr(hypothesis.raw)
    sequence = [(method, url) for method, url, _ in client.calls]
    assert sequence == [
        ("POST", "https://api.omi.health/v1/jobs"),
        ("POST", "https://storage.example/bucket"),
        ("POST", "https://api.omi.health/v1/jobs/job_42/complete"),
        ("GET", "https://api.omi.health/v1/jobs/job_42"),
        ("GET", "https://api.omi.health/v1/jobs/job_42"),
        ("GET", "https://api.omi.health/v1/jobs/job_42"),
        ("GET", "https://results.example/job_42.json"),
    ]
    err = capsys.readouterr().err
    assert "omi: job job_42 running" in err
    assert "omi: job job_42 succeeded" in err
    assert err.count("running") == 1  # one line per state change


def test_omi_inline_result_content_and_segments_fallback(monkeypatch: pytest.MonkeyPatch, flac: Path):
    monkeypatch.setenv("OMI_API_KEY", OMI_KEY)
    route = _omi_router(
        [
            {
                "state": "completed",
                "result": {
                    "content": {
                        "segments": [
                            {"start": 0.0, "end": 1.0, "text": " hi "},
                            {"start": 1.0, "end": 2.0, "text": "you"},
                        ]
                    }
                },
            }
        ],
        result_payload={},
    )
    client = FakeClient(route)
    provider = OmiProvider(_config("omi", "omi-general-1"), client=client, sleep=lambda _: None)

    hypothesis = provider.transcribe(flac, language="en")

    assert hypothesis.text == "hi you"
    assert hypothesis.language == "en"
    assert hypothesis.segments[0] == {"start_s": 0.0, "end_s": 1.0, "text": "hi"}
    assert not any(url.startswith("https://results.example") for _, url, _ in client.calls)


def test_omi_inline_result_as_json_string(monkeypatch: pytest.MonkeyPatch, flac: Path):
    monkeypatch.setenv("OMI_API_KEY", OMI_KEY)
    route = _omi_router([{"status": "succeeded", "result": {"content": '{"text": "from string"}'}}], result_payload={})
    provider = OmiProvider(_config("omi", "omi-general-1"), client=FakeClient(route), sleep=lambda _: None)

    assert provider.transcribe(flac, language="en").text == "from string"


def test_omi_failed_job_raises_provider_error(monkeypatch: pytest.MonkeyPatch, flac: Path):
    monkeypatch.setenv("OMI_API_KEY", OMI_KEY)
    route = _omi_router(
        [{"status": "queued"}, {"status": "failed", "error": {"message": "audio too noisy"}}], result_payload={}
    )
    provider = OmiProvider(_config("omi", "omi-general-1"), client=FakeClient(route), sleep=lambda _: None)

    with pytest.raises(ProviderError, match="audio too noisy"):
        provider.transcribe(flac, language="en")


def test_omi_download_from_api_host_sends_auth_and_waits_for_decoration(monkeypatch: pytest.MonkeyPatch, flac: Path):
    monkeypatch.setenv("OMI_API_KEY", OMI_KEY)
    sleeps: list[float] = []
    downloads = [
        {"text": "early", "decoration": {"status": "processing"}},
        {"text": "final", "decoration": {"status": "done"}},
    ]
    base = _omi_router(
        [{"status": "succeeded", "result": {"download_url": "https://api.omi.health/v1/results/job_42"}}],
        result_payload={},
    )

    def route(method: str, url: str, kwargs: dict[str, Any]) -> httpx.Response:
        if method == "GET" and url == "https://api.omi.health/v1/results/job_42":
            assert kwargs["headers"]["Authorization"] == f"Bearer {OMI_KEY}"
            return _response(200, url, json=downloads.pop(0))
        return base(method, url, kwargs)

    provider = OmiProvider(_config("omi", "omi-general-1"), client=FakeClient(route), sleep=sleeps.append)

    assert provider.transcribe(flac, language="en").text == "final"
    assert 20.0 in sleeps
    assert downloads == []


def test_omi_deadline_exceeded_is_retryable(monkeypatch: pytest.MonkeyPatch, flac: Path):
    monkeypatch.setenv("OMI_API_KEY", OMI_KEY)
    clock = iter([0.0, 0.0, 0.0, 5000.0, 5000.0, 5000.0])
    monkeypatch.setattr("ocular_evals.providers.omi.time.monotonic", lambda: next(clock, 5000.0))
    route = _omi_router([{"status": "running"}] * 5, result_payload={})
    provider = OmiProvider(
        _config("omi", "omi-general-1", deadline_s=100), client=FakeClient(route), sleep=lambda _: None
    )

    with pytest.raises(ProviderError) as info:
        provider.transcribe(flac, language="en")
    assert info.value.retryable is True
    assert "did not finish" in str(info.value)


def test_omi_check_reports_missing_key_and_bad_options(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OMI_API_KEY", raising=False)
    problems = OmiProvider(_config("omi", "omi-general-1")).check()
    assert any("OMI_API_KEY" in item for item in problems)

    monkeypatch.setenv("OMI_API_KEY", OMI_KEY)
    assert OmiProvider(_config("omi", "omi-general-1")).check() == []
    problems = OmiProvider(_config("omi", "omi-general-1", poll_wait_s=0, nope=True)).check()
    assert any("poll_wait_s" in item for item in problems)
    assert any("nope" in item for item in problems)


# --- audio ---------------------------------------------------------------------


def test_prepare_upload_wav16k_downmixes_and_resamples(tmp_path: Path):
    src = tmp_path / "stereo48k.flac"
    rate = 48000
    t = np.arange(rate * 2) / rate
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.5 * np.sin(2 * np.pi * 660 * t)
    sf.write(str(src), np.stack([left, right], axis=1).astype(np.float32), rate, format="FLAC")
    assert audio_info(src) == (pytest.approx(2.0), 48000)

    out, content_type = prepare_upload(src, fmt="wav16k", workdir=tmp_path / "work")

    assert content_type == "audio/wav"
    assert out.parent == tmp_path / "work"
    info = sf.info(str(out))
    assert info.channels == 1
    assert info.samplerate == 16000
    assert info.subtype == "PCM_16"
    assert info.duration == pytest.approx(2.0, abs=1e-3)


def test_prepare_upload_flac_passthrough_and_content_types(flac: Path):
    out, content_type = prepare_upload(flac, fmt="flac")
    assert out == flac and content_type == "audio/flac"
    assert content_type_for("x.mp3") == "audio/mpeg"
    assert content_type_for("x.WAV") == "audio/wav"
    assert content_type_for("x.m4a") == "audio/mp4"
    assert content_type_for("x.ogg") == "audio/ogg"
    assert content_type_for("x.webm") == "audio/webm"
    with pytest.raises(ProviderError):
        content_type_for("x.aiff")
    with pytest.raises(ProviderError, match="unknown upload format"):
        prepare_upload(flac, fmt="aiff")


def test_prepare_upload_mp3_without_ffmpeg_is_clear(monkeypatch: pytest.MonkeyPatch, flac: Path, tmp_path: Path):
    monkeypatch.setattr("ocular_evals.providers.audio.shutil.which", lambda _: None)
    with pytest.raises(ProviderError, match="ffmpeg"):
        prepare_upload(flac, fmt="mp3", workdir=tmp_path)


# --- faster-whisper ------------------------------------------------------------


def test_faster_whisper_check_mentions_local_extra_when_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    provider = FasterWhisperProvider(_config("faster-whisper", "tiny", compute_type="int8", device="cpu"))

    problems = provider.check()

    assert any("local" in item and "uv sync --extra local" in item for item in problems)
    with pytest.raises(ProviderError, match="local"):
        provider.transcribe(Path("missing.flac"), language="en")


def test_faster_whisper_check_validates_options_without_dependency(monkeypatch: pytest.MonkeyPatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError("no module named faster_whisper")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    provider = FasterWhisperProvider(
        _config("faster-whisper", "tiny", beam_size=0, temperature=[0.0, 0.2], device="tpu", vad_filter="yes", extra=1)
    )

    problems = provider.check()

    assert any("beam_size" in item for item in problems)
    assert any("device" in item for item in problems)
    assert any("vad_filter" in item for item in problems)
    assert any("unknown option" in item and "extra" in item for item in problems)
    assert not any(item.startswith("temperature") for item in problems)


def test_faster_whisper_transcribe_with_fake_model(monkeypatch: pytest.MonkeyPatch, flac: Path):
    class Segment:
        def __init__(self, start: float, end: float, text: str) -> None:
            self.start, self.end, self.text = start, end, text

    class Info:
        language = "en"
        language_probability = 0.98
        duration = 2.5

    captured: dict[str, Any] = {}

    class WhisperModel:
        def __init__(self, model: str, **kwargs: Any) -> None:
            if "revision" in kwargs:
                raise TypeError("unexpected keyword argument 'revision'")
            captured["init"] = {"model": model, **kwargs}

        def transcribe(self, path: str, **options: Any):
            captured["options"] = options
            return iter([Segment(0.0, 1.0, " Hello "), Segment(1.0, 2.5, "there. ")]), Info()

    import types

    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = WhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)

    provider = FasterWhisperProvider(
        _config(
            "faster-whisper",
            "tiny",
            compute_type="int8",
            device="cpu",
            revision="abc123",
            vad_min_silence_ms=500,
            best_of=3,
        )
    )
    assert provider.check() == []

    hypothesis = provider.transcribe(flac, language="en-US")

    assert hypothesis.text == "Hello there."
    assert hypothesis.language == "en"
    assert hypothesis.duration_s == 2.5
    assert hypothesis.segments == (
        {"start_s": 0.0, "end_s": 1.0, "text": "Hello"},
        {"start_s": 1.0, "end_s": 2.5, "text": "there."},
    )
    assert hypothesis.raw["language_probability"] == 0.98
    assert captured["init"] == {"model": "tiny", "device": "cpu", "compute_type": "int8"}
    assert captured["options"]["language"] == "en"
    assert captured["options"]["beam_size"] == 5
    assert captured["options"]["best_of"] == 3
    assert captured["options"]["condition_on_previous_text"] is False
    assert captured["options"]["vad_filter"] is True
    assert captured["options"]["vad_parameters"] == {"min_silence_duration_ms": 500}
    # The model is loaded once and reused.
    provider.transcribe(flac, language="en")
    assert captured["init"]["model"] == "tiny"


# --- nemotron-asr ---------------------------------------------------------------


def test_nemotron_check_mentions_extra_when_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    provider = NemotronAsrProvider(_config("nemotron-asr", "nvidia/nemotron-3.5-asr-streaming-0.6b", device="cpu"))

    problems = provider.check()

    assert any("uv sync --extra nemotron" in item for item in problems)
    with pytest.raises(ProviderError, match="nemotron"):
        provider.transcribe(Path("missing.flac"), language="en")


def test_nemotron_check_validates_options():
    provider = NemotronAsrProvider(
        _config(
            "nemotron-asr",
            "nvidia/nemotron-3.5-asr-streaming-0.6b",
            device="tpu",
            lookahead_frames=-1,
            max_chunk_s=900,
            search_s="ten",
            extra=1,
        )
    )

    problems = [item for item in provider.check() if "uv sync" not in item]

    assert any("device" in item for item in problems)
    assert any("lookahead_frames" in item for item in problems)
    assert any("max_chunk_s" in item and "400" in item for item in problems)
    assert any("search_s" in item for item in problems)
    assert any("unknown option" in item and "extra" in item for item in problems)
    clean = NemotronAsrProvider(
        _config("nemotron-asr", "nvidia/nemotron-speech-streaming-en-0.6b", device="cpu", lookahead_frames=13)
    )
    assert [item for item in clean.check() if "uv sync" not in item] == []


def test_nemotron_transcribe_with_fake_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    captured: dict[str, Any] = {"calls": []}

    class Output:
        sequences = [[1, 2, 3]]

    class Processor:
        supported_num_lookahead_tokens = [13, 6, 1, 0]
        default_num_lookahead_tokens = 13

        def set_num_lookahead_tokens(self, value: int) -> None:
            self.default_num_lookahead_tokens = value

        def __call__(self, audio, sampling_rate, return_tensors, language=None):
            captured["calls"].append({"samples": int(np.asarray(audio).size), "language": language})
            return self

        def to(self, device):
            captured["device"] = device
            return {}

        def decode(self, sequences, skip_special_tokens):
            return [f"chunk {len(captured['calls'])}"]

    class Model:
        def to(self, device):
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            return Output()

    class Torch:
        class backends:
            class mps:
                @staticmethod
                def is_available() -> bool:
                    return False

        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

        @staticmethod
        def inference_mode():
            import contextlib

            return contextlib.nullcontext()

    monkeypatch.setattr(
        "ocular_evals.providers.nemotron_asr._import_transformers",
        lambda: (
            Torch,
            type("M", (), {"from_pretrained": staticmethod(lambda *a, **k: Model())}),
            type("P", (), {"from_pretrained": staticmethod(lambda *a, **k: Processor())}),
        ),
    )
    # 25 s of silence with a 10 s chunk limit -> 3 chunks.
    path = tmp_path / "long.flac"
    sf.write(str(path), np.zeros((25 * 16000, 1), dtype=np.float32), 16000, format="FLAC")
    provider = NemotronAsrProvider(
        _config(
            "nemotron-asr", "nvidia/nemotron-3.5-asr-streaming-0.6b", lookahead_frames=6, max_chunk_s=10, search_s=1
        )
    )

    hypothesis = provider.transcribe(path, language="en")

    assert hypothesis.text == "chunk 1 chunk 2 chunk 3"
    assert len(hypothesis.segments) == 3 and hypothesis.segments[0]["start_s"] == 0.0
    assert all(call["language"] == "en-US" for call in captured["calls"])
    assert captured["device"] == "cpu"
    assert hypothesis.raw["lookahead_frames"] == 6 and hypothesis.raw["chunk_ms"] == 560
    assert hypothesis.language == "en-US" and hypothesis.duration_s == pytest.approx(25.0)


def test_nemotron_rejects_unsupported_lookahead(monkeypatch: pytest.MonkeyPatch):
    class Processor:
        supported_num_lookahead_tokens = [13, 6, 1, 0]

        def set_num_lookahead_tokens(self, value: int) -> None:
            raise AssertionError("should not be reached")

    monkeypatch.setattr(
        "ocular_evals.providers.nemotron_asr._import_transformers",
        lambda: (
            None,
            type("M", (), {"from_pretrained": staticmethod(lambda *a, **k: object())}),
            type("P", (), {"from_pretrained": staticmethod(lambda *a, **k: Processor())}),
        ),
    )
    provider = NemotronAsrProvider(
        _config("nemotron-asr", "nvidia/nemotron-speech-streaming-en-0.6b", lookahead_frames=3)
    )
    with pytest.raises(ProviderError, match="supports lookahead_frames"):
        provider._load()


# --- openai-realtime ------------------------------------------------------------


def test_openai_realtime_check_and_session(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    good = OpenAIRealtimeProvider(_config("openai-realtime", "whisper-1", silence_ms=700, vad_threshold=0.6))
    assert good.check() == []
    session = good._session_update("en-US")["session"]
    assert session["type"] == "transcription"
    assert session["audio"]["input"]["transcription"] == {"model": "whisper-1", "language": "en"}
    assert session["audio"]["input"]["turn_detection"] == {
        "type": "server_vad",
        "silence_duration_ms": 700,
        "threshold": 0.6,
    }
    assert session["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}

    bad = OpenAIRealtimeProvider(
        _config(
            "openai-realtime", "whisper-1", speed=0, vad_threshold=2, prefix_padding_ms=-1, language_hint="yes", x=1
        )
    )
    problems = bad.check()
    assert any("speed" in item for item in problems)
    assert any("vad_threshold" in item for item in problems)
    assert any("prefix_padding_ms" in item for item in problems)
    assert any("language_hint" in item for item in problems)
    assert any("unknown option" in item for item in problems)

    monkeypatch.delenv("OPENAI_API_KEY")
    assert any("OPENAI_API_KEY" in item for item in good.check())


def test_openai_realtime_pcm_is_24k_mono_int16(tmp_path: Path):
    path = tmp_path / "tone.flac"
    tone = (0.5 * np.sin(np.linspace(0, 2 * np.pi * 440, 48000, endpoint=False))).astype(np.float32)
    sf.write(str(path), np.stack([tone, tone], axis=1), 48000, format="FLAC")
    pcm = _load_pcm24k(path)
    assert len(pcm) == 24000 * 2  # one second at 24 kHz, 16-bit
    samples = np.frombuffer(pcm, dtype="<i2")
    assert 15000 < samples.max() <= 16384


def test_openai_realtime_transcribe_joins_turns_in_order(monkeypatch: pytest.MonkeyPatch, flac: Path):
    provider = OpenAIRealtimeProvider(_config("openai-realtime", "whisper-1"))

    async def fake_stream(pcm: bytes, language: str):
        assert language == "en" and len(pcm) > 0
        return {
            "order": ["item_b", "item_a"],
            "transcripts": {"item_a": "second turn.", "item_b": "First turn,"},
            "bounds": {"item_b": {"start_ms": 0, "end_ms": 900}, "item_a": {"start_ms": 1200}},
            "events": {"conversation.item.input_audio_transcription.completed": 2},
        }

    monkeypatch.setattr(provider, "_stream", fake_stream)
    hypothesis = provider.transcribe(flac, language="en")
    assert hypothesis.text == "First turn, second turn."
    assert hypothesis.segments[0] == {"start_s": 0.0, "end_s": 0.9, "text": "First turn,"}
    assert hypothesis.segments[1]["end_s"] is None
    assert hypothesis.raw["turns"] == 2 and hypothesis.raw["model"] == "whisper-1"
