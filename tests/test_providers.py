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
from ocular_evals.providers.audio import audio_info, content_type_for, prepare_upload
from ocular_evals.providers.faster_whisper import FasterWhisperProvider
from ocular_evals.providers.http import error_from_response, send_with_retry
from ocular_evals.providers.omi import OmiProvider
from ocular_evals.providers.openai_compatible import OpenAICompatibleProvider

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
    env = {"omi": ("OMI_API_KEY",), "openai-compatible": ("OPENAI_API_KEY",)}.get(provider, ())
    return ModelConfig(
        name=model, display_name=model, vendor=provider, provider=provider, model=model, env=env, options=options
    )


@pytest.fixture
def flac(tmp_path: Path) -> Path:
    path = tmp_path / "speaker.flac"
    sf.write(str(path), np.zeros((1600, 1), dtype=np.float32), 16000, format="FLAC")
    return path


# --- registry ------------------------------------------------------------------


def test_registry_exposes_the_three_providers():
    assert set(PROVIDERS) == {"openai-compatible", "omi", "faster-whisper"}
    assert isinstance(create_provider(_config("omi", "omi-general-1")), OmiProvider)


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
