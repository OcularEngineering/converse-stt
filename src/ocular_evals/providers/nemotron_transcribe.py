"""NVIDIA Nemotron Transcribe served behind an OpenAI-style ``POST /v1/chat/completions``.

Nemotron 3.5 Transcribe (``nvidia/Nemotron-3.5-Transcribe-30B-A4B``) is an audio-in, text-out
chat model: the recording travels as a base64 ``data:`` URL inside a user message next to a text
instruction, and the transcript comes back as ``choices[0].message.content``. This adapter follows
the model's evaluation guide: no system prompt, the published single-speaker instruction,
reasoning disabled (``chat_template_kwargs.enable_thinking=false``), greedy decoding
(``temperature=0``, ``top_p=1.0``, ``seed=0``) with a 16,384-token completion budget, and the
guide's generation-loop cleanup applied exactly once to the decoded text before it is scored.

The server's base URL is read from an environment variable (default
``NEMOTRON_TRANSCRIBE_ENDPOINT``) so that private deployments never appear in a config file; an
optional bearer token can be named with ``api_key_env``. Audio is sent as mono 16 kHz FLAC by
default (the smallest lossless form the service accepts); ``chunk_s`` cuts long recordings at
pauses when a gateway cannot hold a request open for a whole track.
"""

from __future__ import annotations

import base64
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import httpx

from ocular_evals import __version__
from ocular_evals.config import ModelConfig
from ocular_evals.env import redact
from ocular_evals.providers.audio import audio_info, prepare_upload, split_wav16k
from ocular_evals.providers.base import Hypothesis, Provider, ProviderError, primary_language_tag
from ocular_evals.providers.http import error_from_response, send_with_retry

DEFAULT_ENDPOINT_ENV = "NEMOTRON_TRANSCRIBE_ENDPOINT"
DEFAULT_MAX_TOKENS = 16_384
DEFAULT_TIMEOUT_S = 1_200.0
UPLOADS = ("flac16k", "wav16k", "flac")  # the service accepts WAV and FLAC data URLs

# The evaluation guide's default single-speaker instruction, verbatim.
SINGLE_SPEAKER_PROMPT = (
    "Audio transcription. Language: as spoken. Wording: verbatim, including fillers and repetitions. "
    "Formatting: punctuation and capitalization. Output: transcript only."
)

# Generation-loop cleanup from the evaluation guide: applied once, after decoding, before scoring.
_CHAR_RUN = re.compile(r"(.)\1{20,}", re.DOTALL)
_BLOCK_RUN = re.compile(r"(.{1,4096}?)\1{19,}", re.DOTALL)


def cleanup_generation_loops(text: str) -> str:
    """Collapse a character repeated >= 21 times, then an exact 1-4096-char block repeated >= 20 times."""
    text = _CHAR_RUN.sub(lambda m: m.group(1), text)
    return _BLOCK_RUN.sub(lambda m: m.group(1), text)


def _is_number(value: Any, *, minimum: float, exclusive: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return False
    return value > minimum if exclusive else value >= minimum


class NemotronTranscribeProvider(Provider):
    key = "nemotron-transcribe"
    option_names = frozenset(
        {
            "endpoint_env",
            "api_key_env",
            "upload",
            "timeout_s",
            "prompt",
            "max_tokens",
            "temperature",
            "top_p",
            "seed",
            "loop_cleanup",
            "user_agent",
            "chunk_s",
            "search_s",
        }
    )

    def __init__(self, config: ModelConfig, *, client: httpx.Client | None = None) -> None:
        super().__init__(config)
        options = config.options
        self.endpoint_env = str(options.get("endpoint_env", DEFAULT_ENDPOINT_ENV))
        self.api_key_env = options.get("api_key_env")
        self.upload = str(options.get("upload", "flac16k"))
        self.timeout_s = float(options.get("timeout_s", DEFAULT_TIMEOUT_S))
        self.prompt = str(options.get("prompt", SINGLE_SPEAKER_PROMPT))
        self.max_tokens = options.get("max_tokens", DEFAULT_MAX_TOKENS)
        self.temperature = options.get("temperature", 0)
        self.top_p = options.get("top_p", 1.0)
        self.seed = options.get("seed", 0)
        self.loop_cleanup = options.get("loop_cleanup", True)
        self.user_agent = options.get("user_agent")
        self.chunk_s = options.get("chunk_s")
        self.search_s = options.get("search_s", 10.0)
        self._client = client
        self._owns_client = client is None
        self._workdir: Path | None = None

    def required_env(self) -> tuple[str, ...]:
        names = [self.endpoint_env]
        if self.api_key_env:
            names.append(str(self.api_key_env))
        return tuple(names)

    def check(self) -> list[str]:
        problems = super().check()
        if self.upload not in UPLOADS:
            problems.append(f"upload must be one of {', '.join(UPLOADS)}; got {self.upload!r}")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            problems.append("timeout_s must be a positive number")
        if not self.prompt.strip():
            problems.append("prompt must not be empty")
        if not (isinstance(self.max_tokens, int) and not isinstance(self.max_tokens, bool) and self.max_tokens > 0):
            problems.append("max_tokens must be a positive integer")
        if not _is_number(self.temperature, minimum=0):
            problems.append("temperature must be a number >= 0")
        if not _is_number(self.top_p, minimum=0, exclusive=True) or self.top_p > 1:
            problems.append("top_p must be a number in (0, 1]")
        if self.seed is not None and not (isinstance(self.seed, int) and not isinstance(self.seed, bool)):
            problems.append("seed must be an integer")
        if not isinstance(self.loop_cleanup, bool):
            problems.append("loop_cleanup must be true or false")
        for name in ("api_key_env", "user_agent"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                problems.append(f"{name} must be a non-empty string")
        if self.chunk_s is not None and not _is_number(self.chunk_s, minimum=0, exclusive=True):
            problems.append("chunk_s must be a positive number of seconds")
        if not _is_number(self.search_s, minimum=0):
            problems.append("search_s must be a number of seconds >= 0")
        endpoint = os.environ.get(self.endpoint_env)
        if endpoint and not endpoint.startswith(("http://", "https://")):
            problems.append(f"{self.endpoint_env} must start with http:// or https://")
        return problems

    # -- transport -------------------------------------------------------------------------------

    def _endpoint(self) -> str:
        value = os.environ.get(self.endpoint_env, "").strip().rstrip("/")
        if not value:
            raise ProviderError(f"environment variable {self.endpoint_env} is not set")
        return value if value.endswith("/v1/chat/completions") else f"{value}/v1/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": str(self.user_agent) if self.user_agent else f"ocular-evals/{__version__}",
        }
        if self.api_key_env:
            token = os.environ.get(str(self.api_key_env))
            if not token:
                raise ProviderError(f"environment variable {self.api_key_env} is not set")
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=httpx.Timeout(self.timeout_s, connect=30.0), follow_redirects=False)
        return self._client

    def _payload(self, audio_bytes: bytes, content_type: str) -> dict[str, Any]:
        encoded = base64.b64encode(audio_bytes).decode("ascii")
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {"type": "audio_url", "audio_url": {"url": f"data:{content_type};base64,{encoded}"}},
                    ],
                }
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": int(self.max_tokens),
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        return payload

    def _settings(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "system_prompt": None,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "max_tokens": int(self.max_tokens),
            "enable_thinking": False,
            "loop_cleanup": bool(self.loop_cleanup),
        }

    def _request(self, upload_path: Path, content_type: str) -> tuple[str, dict[str, Any]]:
        url = self._endpoint()
        headers = self._headers()
        body = upload_path.read_bytes()
        payload = self._payload(body, content_type)
        client = self._get_client()

        def send() -> httpx.Response:
            return client.post(url, headers=headers, json=payload, timeout=self.timeout_s)

        response = send_with_retry(send)
        if response.status_code != 200:
            raise error_from_response(response, context=f"transcription request to {url} failed")
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"{url} returned a non-JSON body: {response.text[:300]}") from exc
        if not isinstance(data, dict):
            raise ProviderError(f"{url} returned a JSON {type(data).__name__}, expected an object")
        if data.get("error"):
            raise ProviderError(f"{url} returned an error: {str(data['error'])[:300]}")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ProviderError(f"{url} response has no choices")
        choice = choices[0]
        finish = choice.get("finish_reason")
        if finish != "stop":
            raise ProviderError(
                f"{url}: incomplete transcript for {upload_path.name} (finish_reason={finish!r}); "
                "raise max_tokens or set chunk_s"
            )
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        content = message.get("content")
        if not isinstance(content, str):
            raise ProviderError(f"{url} response lacks message.content")
        raw: dict[str, Any] = {
            "finish_reason": finish,
            "usage": data.get("usage"),
            "upload": {"format": self.upload, "content_type": content_type, "bytes": len(body)},
            "response_model": data.get("model"),
        }
        request_id = response.headers.get("x-request-id") or data.get("id")
        if request_id:
            raw["request_id"] = request_id
        return content, raw

    def _finish(self, text: str) -> tuple[str, dict[str, Any]]:
        text = text.strip()
        if not self.loop_cleanup:
            return text, {"applied": False}
        cleaned = cleanup_generation_loops(text)
        return cleaned, {
            "applied": True,
            "changed": cleaned != text,
            "chars_before": len(text),
            "chars_after": len(cleaned),
        }

    # -- public --------------------------------------------------------------------------------

    def transcribe(self, audio: Path, *, language: str) -> Hypothesis:
        if (self.upload != "flac" or self.chunk_s) and self._workdir is None:
            self._workdir = Path(tempfile.mkdtemp(prefix="ocular-evals-"))
        duration_s, _ = audio_info(audio)
        lang = primary_language_tag(language)
        base_raw: dict[str, Any] = {
            "endpoint": self._endpoint(),
            "model": self.config.model,
            "settings": self._settings(),
        }

        if not self.chunk_s:
            upload_path, content_type = prepare_upload(audio, fmt=self.upload, workdir=self._workdir)
            content, raw = self._request(upload_path, content_type)
            if upload_path != audio:
                upload_path.unlink(missing_ok=True)
            text, cleanup = self._finish(content)
            return Hypothesis(
                text=text,
                language=lang,
                duration_s=duration_s,
                raw={**base_raw, **raw, "loop_cleanup": cleanup, "content": content},
            )

        chunks = split_wav16k(
            audio, max_chunk_s=float(self.chunk_s), search_s=float(self.search_s), workdir=self._workdir
        )
        pieces: list[str] = []
        parts: list[dict[str, Any]] = []
        for index, (chunk, start_s) in enumerate(chunks):
            if len(chunks) > 1:
                print(
                    redact(f"{self.key}: {audio.name} chunk {index + 1}/{len(chunks)} from {start_s:.1f}s"),
                    file=sys.stderr,
                )
            fmt = "flac16k" if self.upload == "flac" else self.upload
            upload_path, content_type = prepare_upload(chunk, fmt=fmt, workdir=self._workdir)
            content, raw = self._request(upload_path, content_type)
            text, cleanup = self._finish(content)
            if text:
                pieces.append(text)
            parts.append(
                {
                    "start_s": round(start_s, 3),
                    "bytes": raw["upload"]["bytes"],
                    "finish_reason": raw["finish_reason"],
                    "usage": raw.get("usage"),
                    "loop_cleanup": cleanup,
                    "content": content,
                }
            )
            if upload_path != chunk:
                upload_path.unlink(missing_ok=True)
            chunk.unlink(missing_ok=True)
        return Hypothesis(
            text=" ".join(pieces).strip(),
            language=lang,
            duration_s=duration_s,
            raw={**base_raw, "upload": {"format": self.upload, "chunk_s": self.chunk_s, "chunks": parts}},
        )

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None
        if self._workdir is not None:
            shutil.rmtree(self._workdir, ignore_errors=True)
            self._workdir = None


__all__ = ["NemotronTranscribeProvider", "SINGLE_SPEAKER_PROMPT", "cleanup_generation_loops"]
