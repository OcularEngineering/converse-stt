"""Meta Model API speech-to-text: ``POST /v1/asr/transcribe`` (Muse Voice Transcribe).

The endpoint takes a multipart body with two parts: ``request`` (JSON settings) and ``audio``
(a mono 16-bit PCM WAV at 16 or 24 kHz, at most 10 minutes and 32 MB). Ocular AI tracks run
about 15 minutes, so each track is cut into chunks of ``chunk_s`` seconds at the quietest point
near each even split, every chunk is transcribed, and the transcripts are joined in order.

``mode`` is ``ENDPOINTING`` by default (model-detected speech turns, which suits long
multi-utterance recordings); ``PUSH_TO_TALK`` treats a whole clip as one turn.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import httpx

from ocular_evals.config import ModelConfig
from ocular_evals.env import redact
from ocular_evals.providers.audio import split_wav16k
from ocular_evals.providers.base import Hypothesis, Provider, ProviderError, primary_language_tag
from ocular_evals.providers.http import error_from_response, send_with_retry

DEFAULT_BASE_URL = "https://api.meta.ai/v1"
DEFAULT_API_KEY_ENV = "META_API_KEY"
MODES = ("PUSH_TO_TALK", "ENDPOINTING", "DIARIZATION")
MAX_CLIP_S = 600.0  # the API rejects clips over 10 minutes
DEFAULT_CHUNK_S = 540.0


class MetaProvider(Provider):
    key = "meta"
    option_names = frozenset({"base_url", "api_key_env", "timeout_s", "mode", "chunk_s", "search_s"})

    def __init__(self, config: ModelConfig, *, client: httpx.Client | None = None) -> None:
        super().__init__(config)
        options = config.options
        self.base_url = str(options.get("base_url", DEFAULT_BASE_URL)).rstrip("/")
        self.api_key_env = str(options.get("api_key_env", DEFAULT_API_KEY_ENV))
        self.timeout_s = float(options.get("timeout_s", 600))
        self.mode = str(options.get("mode", "ENDPOINTING")).upper()
        self.chunk_s = float(options.get("chunk_s", DEFAULT_CHUNK_S))
        self.search_s = float(options.get("search_s", 30.0))
        self._client = client
        self._owns_client = client is None
        self._workdir: Path | None = None

    def required_env(self) -> tuple[str, ...]:
        return (self.api_key_env,)

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/asr/transcribe"

    def check(self) -> list[str]:
        problems = super().check()
        if self.mode not in MODES:
            problems.append(f"mode must be one of {', '.join(MODES)}; got {self.mode!r}")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            problems.append("timeout_s must be a positive number")
        if not math.isfinite(self.chunk_s) or not 0 < self.chunk_s <= MAX_CLIP_S:
            problems.append(f"chunk_s must be between 0 and {MAX_CLIP_S:.0f} (the API's 10-minute clip limit)")
        if not math.isfinite(self.search_s) or self.search_s < 0:
            problems.append("search_s must be zero or a positive number")
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

    def _request_body(self) -> dict[str, Any]:
        return {"model": self.config.model, "audioEncoding": "WAV", "mode": self.mode}

    def _transcribe_chunk(self, chunk: Path, *, headers: dict[str, str], request: dict[str, Any]) -> dict[str, Any]:
        client = self._get_client()
        url = self.endpoint
        request_json = json.dumps(request, separators=(",", ":"))

        def send() -> httpx.Response:
            with chunk.open("rb") as handle:
                return client.post(
                    url,
                    headers=headers,
                    files={
                        "request": (None, request_json, "application/json"),
                        "audio": (chunk.name, handle, "audio/wav"),
                    },
                    timeout=self.timeout_s,
                )

        response = send_with_retry(send)
        if response.status_code != 200:
            raise error_from_response(response, context=f"transcription request to {url} failed")
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"{url} returned a non-JSON body: {response.text[:300]}") from exc
        if not isinstance(data, dict):
            raise ProviderError(f"{url} returned a JSON {type(data).__name__}, expected an object")
        return data

    def transcribe(self, audio: Path, *, language: str) -> Hypothesis:
        if self._workdir is None:
            self._workdir = Path(tempfile.mkdtemp(prefix="ocular-evals-"))
        chunks = split_wav16k(audio, max_chunk_s=self.chunk_s, search_s=self.search_s, workdir=self._workdir)
        headers = {"Authorization": f"Bearer {self._api_key()}", "Accept": "application/json"}
        request = self._request_body()
        language = primary_language_tag(language)
        texts: list[str] = []
        segments: list[dict] = []
        parts: list[dict[str, Any]] = []
        total_ms = 0.0
        for index, (chunk, start_s) in enumerate(chunks):
            if len(chunks) > 1:
                note = f"meta: {audio.name} chunk {index + 1}/{len(chunks)} from {start_s:.1f}s"
                print(redact(note), file=sys.stderr, flush=True)
            data = self._transcribe_chunk(chunk, headers=headers, request=request)
            transcript = data.get("transcript")
            turns = data.get("turns") if isinstance(data.get("turns"), list) else []
            if not isinstance(transcript, str):
                turn_texts = [str(turn.get("transcript") or "") for turn in turns if isinstance(turn, dict)]
                if not turn_texts:
                    raise ProviderError(f"{self.endpoint} response lacks `transcript` and `turns`")
                transcript = " ".join(text for text in turn_texts if text)
            if transcript.strip():
                texts.append(transcript.strip())
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                start_ms, end_ms = turn.get("startMs"), turn.get("endMs")
                segments.append(
                    {
                        "start_s": start_s + float(start_ms) / 1000.0 if isinstance(start_ms, (int, float)) else None,
                        "end_s": start_s + float(end_ms) / 1000.0 if isinstance(end_ms, (int, float)) else None,
                        "text": str(turn.get("transcript") or "").strip(),
                    }
                )
            duration_ms = data.get("audioDurationMs")
            if isinstance(duration_ms, (int, float)) and not isinstance(duration_ms, bool):
                total_ms += float(duration_ms)
            parts.append(
                {
                    "start_s": round(start_s, 3),
                    "bytes": chunk.stat().st_size,
                    "session_id": data.get("sessionId"),
                    "turns": len(turns),
                }
            )
            chunk.unlink(missing_ok=True)
        raw = {
            "endpoint": self.endpoint,
            "parameters": {**request, "language": language},
            "upload": {"format": "wav16k", "content_type": "audio/wav", "chunks": parts},
        }
        return Hypothesis(
            text=" ".join(texts).strip(),
            language=language,
            duration_s=total_ms / 1000.0 if total_ms else None,
            segments=tuple(segments),
            raw=raw,
        )

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None
        if self._workdir is not None:
            shutil.rmtree(self._workdir, ignore_errors=True)
            self._workdir = None


__all__ = ["MetaProvider"]
