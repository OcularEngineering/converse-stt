"""Synchronous ``POST /audio/transcriptions`` providers (OpenAI and compatible servers)."""

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

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
RESPONSE_FORMATS = ("json", "verbose_json", "text")


class OpenAICompatibleProvider(Provider):
    key = "openai-compatible"
    option_names = frozenset(
        {
            "base_url",
            "api_key_env",
            "response_format",
            "upload",
            "timeout_s",
            "temperature",
            "prompt",
            "send_language",
        }
    )

    def __init__(self, config: ModelConfig, *, client: httpx.Client | None = None) -> None:
        super().__init__(config)
        options = config.options
        self.base_url = str(options.get("base_url", DEFAULT_BASE_URL)).rstrip("/")
        self.api_key_env = str(options.get("api_key_env", DEFAULT_API_KEY_ENV))
        self.response_format = str(options.get("response_format", "json"))
        self.upload = str(options.get("upload", "flac"))
        self.timeout_s = float(options.get("timeout_s", 600))
        self.temperature = options.get("temperature")
        self.prompt = options.get("prompt")
        self.send_language = bool(options.get("send_language", True))
        self._client = client
        self._owns_client = client is None
        self._workdir: Path | None = None

    def required_env(self) -> tuple[str, ...]:
        return (self.api_key_env,)

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/audio/transcriptions"

    def check(self) -> list[str]:
        problems = super().check()
        if self.response_format not in RESPONSE_FORMATS:
            problems.append(
                f"response_format must be one of {', '.join(RESPONSE_FORMATS)}; got {self.response_format!r}"
            )
        if self.upload not in UPLOAD_FORMATS:
            problems.append(f"upload must be one of {', '.join(UPLOAD_FORMATS)}; got {self.upload!r}")
        elif self.upload == "mp3" and not ffmpeg_available():
            problems.append('upload = "mp3" requires an ffmpeg binary on PATH')
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            problems.append("timeout_s must be a positive number")
        if self.temperature is not None and (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
            or not 0 <= self.temperature <= 1
        ):
            problems.append("temperature must be a number between 0 and 1")
        if self.prompt is not None and not isinstance(self.prompt, str):
            problems.append("prompt must be a string")
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
        parameters: dict[str, Any] = {"model": self.config.model, "response_format": self.response_format}
        if self.send_language:
            parameters["language"] = primary_language_tag(language)
        if self.temperature is not None:
            parameters["temperature"] = self.temperature
        if self.prompt:
            parameters["prompt"] = self.prompt
        return parameters

    def transcribe(self, audio: Path, *, language: str) -> Hypothesis:
        if self.upload != "flac" and self._workdir is None:
            self._workdir = Path(tempfile.mkdtemp(prefix="ocular-evals-"))
        upload_path, content_type = prepare_upload(audio, fmt=self.upload, workdir=self._workdir)
        parameters = self._parameters(language)
        form = {key: str(value) for key, value in parameters.items()}
        headers = {"Authorization": f"Bearer {self._api_key()}"}
        client = self._get_client()
        url = self.endpoint

        def send() -> httpx.Response:
            with upload_path.open("rb") as handle:
                return client.post(
                    url,
                    headers=headers,
                    data=form,
                    files={"file": (upload_path.name, handle, content_type)},
                    timeout=self.timeout_s,
                )

        response = send_with_retry(send)
        raw = {
            "endpoint": url,
            "parameters": parameters,
            "upload": {"format": self.upload, "content_type": content_type, "bytes": upload_path.stat().st_size},
        }
        if response.status_code == 202:
            raise ProviderError(
                f"{url} answered HTTP 202: the server queued an asynchronous job instead of returning a transcript. "
                "Use the `omi` provider for long audio handled through a job API.",
                retryable=False,
                status=202,
            )
        if response.status_code != 200:
            raise error_from_response(response, context=f"transcription request to {url} failed")
        request_id = response.headers.get("x-request-id")
        if request_id:
            raw["request_id"] = request_id
        if self.response_format == "text":
            return Hypothesis(text=response.text.strip(), language=parameters.get("language"), raw=raw)
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"{url} returned a non-JSON body: {response.text[:300]}") from exc
        if not isinstance(data, dict):
            raise ProviderError(f"{url} returned a JSON {type(data).__name__}, expected an object")
        segments = coerce_segments(data.get("segments"))
        text = data.get("text")
        if not isinstance(text, str):
            if segments:
                text = " ".join(segment["text"] for segment in segments if segment["text"])
            else:
                raise ProviderError(f"{url} response lacks a `text` field")
        duration = data.get("duration")
        return Hypothesis(
            text=text.strip(),
            language=data.get("language") if isinstance(data.get("language"), str) else parameters.get("language"),
            duration_s=float(duration)
            if isinstance(duration, (int, float)) and not isinstance(duration, bool)
            else None,
            segments=segments,
            raw=raw,
        )

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None
        if self._workdir is not None:
            shutil.rmtree(self._workdir, ignore_errors=True)
            self._workdir = None


__all__ = ["OpenAICompatibleProvider"]
