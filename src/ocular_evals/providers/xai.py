"""xAI speech-to-text: ``POST /v1/stt`` (Grok Voice Transcribe).

The endpoint takes the same kind of multipart form as the OpenAI protocol but lives at
``/stt`` rather than ``/audio/transcriptions``, and its knobs differ: ``format`` turns on
inverse text normalization (spoken numbers -> digits) and ``filler_words`` keeps "uh" / "um"
in the transcript. The harness scores against verbatim references, so filler words are kept
by default. The response is ``{"text", "language", "duration", "words": [...]}``.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import httpx

from ocular_evals.config import ModelConfig
from ocular_evals.providers.audio import UPLOAD_FORMATS, ffmpeg_available, prepare_upload
from ocular_evals.providers.base import Hypothesis, Provider, ProviderError, coerce_segments, primary_language_tag
from ocular_evals.providers.http import error_from_response, send_with_retry

DEFAULT_BASE_URL = "https://api.x.ai/v1"
DEFAULT_API_KEY_ENV = "XAI_API_KEY"


def _form_value(value: Any) -> str:
    """Multipart form fields are strings; booleans go as the lowercase ``true`` / ``false`` the API expects."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


class XaiProvider(Provider):
    key = "xai"
    option_names = frozenset(
        {"base_url", "api_key_env", "upload", "timeout_s", "format", "filler_words", "send_language"}
    )

    def __init__(self, config: ModelConfig, *, client: httpx.Client | None = None) -> None:
        super().__init__(config)
        options = config.options
        self.base_url = str(options.get("base_url", DEFAULT_BASE_URL)).rstrip("/")
        self.api_key_env = str(options.get("api_key_env", DEFAULT_API_KEY_ENV))
        self.upload = str(options.get("upload", "flac"))
        self.timeout_s = float(options.get("timeout_s", 600))
        self.format = options.get("format", True)
        self.filler_words = options.get("filler_words", True)
        self.send_language = options.get("send_language", True)
        self._client = client
        self._owns_client = client is None
        self._workdir: Path | None = None

    def required_env(self) -> tuple[str, ...]:
        return (self.api_key_env,)

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/stt"

    def check(self) -> list[str]:
        problems = super().check()
        if self.upload not in UPLOAD_FORMATS:
            problems.append(f"upload must be one of {', '.join(UPLOAD_FORMATS)}; got {self.upload!r}")
        elif self.upload == "mp3" and not ffmpeg_available():
            problems.append('upload = "mp3" requires an ffmpeg binary on PATH')
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            problems.append("timeout_s must be a positive number")
        for name, value in (("format", self.format), ("filler_words", self.filler_words)):
            if not isinstance(value, bool):
                problems.append(f"{name} must be true or false")
        if not isinstance(self.send_language, bool):
            problems.append("send_language must be true or false")
        if not self.base_url.startswith(("http://", "https://")):
            problems.append(f"base_url must start with http:// or https://; got {self.base_url!r}")
        return problems

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=httpx.Timeout(self.timeout_s, connect=30.0))
        return self._client

    def _api_key(self) -> str:
        value = os.environ.get(self.api_key_env)
        if not value:
            raise ProviderError(f"environment variable {self.api_key_env} is not set")
        return value

    def _parameters(self, language: str) -> dict[str, Any]:
        parameters: dict[str, Any] = {"model": self.config.model}
        if self.send_language:
            parameters["language"] = primary_language_tag(language)
        parameters["format"] = bool(self.format)
        parameters["filler_words"] = bool(self.filler_words)
        return parameters

    def transcribe(self, audio: Path, *, language: str) -> Hypothesis:
        if self.upload != "flac" and self._workdir is None:
            self._workdir = Path(tempfile.mkdtemp(prefix="ocular-evals-"))
        upload_path, content_type = prepare_upload(audio, fmt=self.upload, workdir=self._workdir)
        parameters = self._parameters(language)
        form = {key: _form_value(value) for key, value in parameters.items()}
        headers = {"Authorization": f"Bearer {self._api_key()}"}
        client = self._get_client()
        url = self.endpoint

        def send() -> httpx.Response:
            with upload_path.open("rb") as handle:
                # ``file`` must be the last field in the multipart body; httpx writes data fields first.
                return client.post(
                    url,
                    headers=headers,
                    data=form,
                    files={"file": (upload_path.name, handle, content_type)},
                    timeout=self.timeout_s,
                )

        response = send_with_retry(send)
        if response.status_code != 200:
            raise error_from_response(response, context=f"transcription request to {url} failed")
        raw: dict[str, Any] = {
            "endpoint": url,
            "parameters": parameters,
            "upload": {"format": self.upload, "content_type": content_type, "bytes": upload_path.stat().st_size},
        }
        request_id = response.headers.get("x-request-id")
        if request_id:
            raw["request_id"] = request_id
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"{url} returned a non-JSON body: {response.text[:300]}") from exc
        if not isinstance(data, dict):
            raise ProviderError(f"{url} returned a JSON {type(data).__name__}, expected an object")
        words = coerce_segments(data.get("words"))
        text = data.get("text")
        if not isinstance(text, str):
            if words:
                text = " ".join(word["text"] for word in words if word["text"])
            else:
                raise ProviderError(f"{url} response lacks a `text` field")
        duration = data.get("duration")
        detected = data.get("language")
        return Hypothesis(
            text=text.strip(),
            language=detected if isinstance(detected, str) and detected else parameters.get("language"),
            duration_s=float(duration)
            if isinstance(duration, (int, float)) and not isinstance(duration, bool)
            else None,
            segments=words,
            raw=raw,
        )

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None
        if self._workdir is not None:
            shutil.rmtree(self._workdir, ignore_errors=True)
            self._workdir = None


__all__ = ["XaiProvider"]
