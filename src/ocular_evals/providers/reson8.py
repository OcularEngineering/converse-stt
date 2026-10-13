"""Reson8 speech-to-text: ``POST /v1/speech-to-text/prerecorded``.

The whole audio file goes as the request body (``Content-Type: application/octet-stream``;
the server detects WAV/MP3/FLAC from the container header) and every setting is a query
parameter. There is no model id: requests always run on the current pipeline, so the config's
``model`` is a label only. ``filler_mode=verbatim`` keeps "uh" / "um" in the transcript, which
matches the verbatim references the harness scores against. Authentication is
``Authorization: ApiKey <key>``. The response is ``{"text", "language"?, "words"?: [...]}``
where each word carries ``start_ms`` / ``duration_ms`` when timestamps are requested.
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
from ocular_evals.providers.base import Hypothesis, Provider, ProviderError, primary_language_tag
from ocular_evals.providers.http import error_from_response, send_with_retry

DEFAULT_BASE_URL = "https://api.reson8.dev/v1"
DEFAULT_API_KEY_ENV = "RESON8_API_KEY"
FILLER_MODES = ("clean", "natural", "verbatim")


def _query_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _words_to_segments(items: object) -> tuple[dict, ...]:
    """``{"text", "start_ms", "duration_ms"}`` -> ``{"start_s", "end_s", "text"}``."""
    if not isinstance(items, list):
        return ()
    shaped: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        start = item.get("start_ms")
        duration = item.get("duration_ms")
        has_start = isinstance(start, (int, float)) and not isinstance(start, bool)
        has_duration = isinstance(duration, (int, float)) and not isinstance(duration, bool)
        shaped.append(
            {
                "start_s": float(start) / 1000.0 if has_start else None,
                "end_s": float(start + duration) / 1000.0 if has_start and has_duration else None,
                "text": str(item.get("text") or "").strip(),
            }
        )
    return tuple(shaped)


class Reson8Provider(Provider):
    key = "reson8"
    option_names = frozenset(
        {
            "base_url",
            "api_key_env",
            "upload",
            "timeout_s",
            "filler_mode",
            "send_language",
            "include_words",
            "custom_model_id",
        }
    )

    def __init__(self, config: ModelConfig, *, client: httpx.Client | None = None) -> None:
        super().__init__(config)
        options = config.options
        self.base_url = str(options.get("base_url", DEFAULT_BASE_URL)).rstrip("/")
        self.api_key_env = str(options.get("api_key_env", DEFAULT_API_KEY_ENV))
        self.upload = str(options.get("upload", "flac"))
        self.timeout_s = float(options.get("timeout_s", 900))
        self.filler_mode = str(options.get("filler_mode", "verbatim"))
        self.send_language = options.get("send_language", True)
        self.include_words = options.get("include_words", True)
        self.custom_model_id = options.get("custom_model_id")
        self._client = client
        self._owns_client = client is None
        self._workdir: Path | None = None

    def required_env(self) -> tuple[str, ...]:
        return (self.api_key_env,)

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/speech-to-text/prerecorded"

    def check(self) -> list[str]:
        problems = super().check()
        if self.upload not in UPLOAD_FORMATS:
            problems.append(f"upload must be one of {', '.join(UPLOAD_FORMATS)}; got {self.upload!r}")
        elif self.upload == "mp3" and not ffmpeg_available():
            problems.append('upload = "mp3" requires an ffmpeg binary on PATH')
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            problems.append("timeout_s must be a positive number")
        if self.filler_mode not in FILLER_MODES:
            problems.append(f"filler_mode must be one of {', '.join(FILLER_MODES)}; got {self.filler_mode!r}")
        for name, value in (("send_language", self.send_language), ("include_words", self.include_words)):
            if not isinstance(value, bool):
                problems.append(f"{name} must be true or false")
        if self.custom_model_id is not None and not isinstance(self.custom_model_id, str):
            problems.append("custom_model_id must be a string")
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
        parameters: dict[str, Any] = {"encoding": "auto", "filler_mode": self.filler_mode}
        if self.send_language:
            parameters["language"] = primary_language_tag(language)
        parameters["include_language"] = True
        if self.include_words:
            parameters["include_words"] = True
            parameters["include_timestamps"] = True
        if self.custom_model_id:
            parameters["custom_model_id"] = self.custom_model_id
        return parameters

    def transcribe(self, audio: Path, *, language: str) -> Hypothesis:
        if self.upload != "flac" and self._workdir is None:
            self._workdir = Path(tempfile.mkdtemp(prefix="ocular-evals-"))
        upload_path, content_type = prepare_upload(audio, fmt=self.upload, workdir=self._workdir)
        parameters = self._parameters(language)
        params = {key: _query_value(value) for key, value in parameters.items()}
        headers = {"Authorization": f"ApiKey {self._api_key()}", "Content-Type": "application/octet-stream"}
        client = self._get_client()
        url = self.endpoint
        body = upload_path.read_bytes()

        def send() -> httpx.Response:
            return client.post(url, headers=headers, params=params, content=body, timeout=self.timeout_s)

        response = send_with_retry(send)
        if response.status_code != 200:
            raise error_from_response(response, context=f"transcription request to {url} failed")
        raw: dict[str, Any] = {
            "endpoint": url,
            "parameters": parameters,
            "upload": {"format": self.upload, "content_type": content_type, "bytes": len(body)},
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
        raw["response"] = data
        text = data.get("text")
        words = _words_to_segments(data.get("words"))
        if not isinstance(text, str):
            if words:
                text = " ".join(word["text"] for word in words if word["text"])
            else:
                raise ProviderError(f"{url} response lacks a `text` field")
        detected = data.get("language")
        duration_s = None
        start_ms, duration_ms = data.get("start_ms"), data.get("duration_ms")
        if isinstance(duration_ms, (int, float)) and not isinstance(duration_ms, bool):
            offset = start_ms if isinstance(start_ms, (int, float)) and not isinstance(start_ms, bool) else 0
            duration_s = float(offset + duration_ms) / 1000.0
        return Hypothesis(
            text=text.strip(),
            language=detected if isinstance(detected, str) and detected else parameters.get("language"),
            duration_s=duration_s,
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


__all__ = ["Reson8Provider"]
