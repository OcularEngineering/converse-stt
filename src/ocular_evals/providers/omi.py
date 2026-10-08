"""Omi long-audio transcription through the asynchronous job API."""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from ocular_evals.config import ModelConfig
from ocular_evals.env import redact
from ocular_evals.providers.audio import UPLOAD_FORMATS, ffmpeg_available, prepare_upload
from ocular_evals.providers.base import Hypothesis, Provider, ProviderError, coerce_segments, primary_language_tag
from ocular_evals.providers.http import error_from_response, send_with_retry

DEFAULT_BASE_URL = "https://api.omi.health"
DEFAULT_API_KEY_ENV = "OMI_API_KEY"
API_HOSTS = frozenset({"api.omi.health", "api.eu.omi.health"})
SUCCEEDED = frozenset({"succeeded", "completed"})
FAILED = frozenset({"failed", "cancelled", "canceled", "error"})
PENDING = frozenset({"awaiting_upload", "queued", "accepted", "running", "processing"})


class OmiProvider(Provider):
    key = "omi"
    option_names = frozenset(
        {"base_url", "api_key_env", "response_format", "poll_wait_s", "deadline_s", "timeout_s", "upload"}
    )

    def __init__(self, config: ModelConfig, *, client: httpx.Client | None = None, sleep=time.sleep) -> None:
        super().__init__(config)
        options = config.options
        self.base_url = str(options.get("base_url", DEFAULT_BASE_URL)).rstrip("/")
        self.api_key_env = str(options.get("api_key_env", DEFAULT_API_KEY_ENV))
        self.response_format = str(options.get("response_format", "json"))
        self.poll_wait_s = float(options.get("poll_wait_s", 20))
        self.deadline_s = float(options.get("deadline_s", 2700))
        self.timeout_s = float(options.get("timeout_s", 120))
        self.upload = str(options.get("upload", "flac"))
        self._client = client
        self._owns_client = client is None
        self._sleep = sleep
        self._workdir: Path | None = None

    def required_env(self) -> tuple[str, ...]:
        return (self.api_key_env,)

    @property
    def jobs_url(self) -> str:
        return f"{self.base_url}/v1/jobs"

    def check(self) -> list[str]:
        problems = super().check()
        if self.response_format != "json":
            problems.append(f'response_format must be "json" for scoring; got {self.response_format!r}')
        if self.upload not in UPLOAD_FORMATS:
            problems.append(f"upload must be one of {', '.join(UPLOAD_FORMATS)}; got {self.upload!r}")
        elif self.upload == "mp3" and not ffmpeg_available():
            problems.append('upload = "mp3" requires an ffmpeg binary on PATH')
        for name, value in (
            ("poll_wait_s", self.poll_wait_s),
            ("deadline_s", self.deadline_s),
            ("timeout_s", self.timeout_s),
        ):
            if not math.isfinite(value) or value <= 0:
                problems.append(f"{name} must be a positive number")
        if not self.base_url.startswith(("http://", "https://")):
            problems.append(f"base_url must start with http:// or https://; got {self.base_url!r}")
        return problems

    # -- plumbing -----------------------------------------------------------------

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=httpx.Timeout(self.timeout_s, connect=30.0))
        return self._client

    def _auth(self) -> dict[str, str]:
        value = os.environ.get(self.api_key_env)
        if not value:
            raise ProviderError(f"environment variable {self.api_key_env} is not set")
        return {"Authorization": f"Bearer {value}"}

    def _absolute(self, url: str) -> str:
        return urljoin(self.base_url + "/", url)

    def _auth_for_download(self, url: str) -> dict[str, str]:
        host = (urlparse(url).hostname or "").lower()
        base_host = (urlparse(self.base_url).hostname or "").lower()
        if host in API_HOSTS or (base_host and host == base_host):
            return self._auth()
        return {}

    @staticmethod
    def _status(job_id: str, state: str) -> None:
        print(redact(f"omi: job {job_id} {state}"), file=sys.stderr, flush=True)

    # -- job flow -----------------------------------------------------------------

    def transcribe(self, audio: Path, *, language: str) -> Hypothesis:
        if self.upload != "flac" and self._workdir is None:
            self._workdir = Path(tempfile.mkdtemp(prefix="ocular-evals-"))
        upload_path, content_type = prepare_upload(audio, fmt=self.upload, workdir=self._workdir)
        size = upload_path.stat().st_size
        lang = primary_language_tag(language)
        parameters = {"model": self.config.model, "language": lang, "response_format": self.response_format}
        client = self._get_client()
        headers = self._auth()
        jobs_url = self.jobs_url

        created = send_with_retry(
            lambda: client.post(
                jobs_url,
                headers=headers,
                json={
                    **parameters,
                    "filename": upload_path.name,
                    "content_type": content_type,
                    "content_length_bytes": size,
                },
                timeout=self.timeout_s,
            ),
            sleep=self._sleep,
        )
        if created.status_code >= 400:
            raise error_from_response(created, context="omi: creating the job failed")
        job = self._json_object(created, context="omi: job creation")
        job_id = str(job.get("job_id") or job.get("id") or "?")
        upload = job.get("upload")
        if not isinstance(upload, dict) or not isinstance(upload.get("url"), str):
            raise ProviderError(f"omi: job {job_id} response lacks an upload slot")
        fields = upload.get("fields") or {}
        if not isinstance(fields, dict):
            raise ProviderError(f"omi: job {job_id} upload fields are not an object")
        complete_url = job.get("complete_url")
        poll_url = job.get("poll_url")
        if not (isinstance(complete_url, str) and complete_url and isinstance(poll_url, str) and poll_url):
            raise ProviderError(f"omi: job {job_id} response lacks complete_url/poll_url")
        complete_url = self._absolute(complete_url)
        poll_url = self._absolute(poll_url)
        upload_url = str(upload["url"])
        self._status(job_id, "uploading")

        def send_upload() -> httpx.Response:
            with upload_path.open("rb") as handle:
                return client.post(
                    upload_url,
                    data={key: str(value) for key, value in fields.items()},
                    files={"file": (upload_path.name, handle, content_type)},
                    timeout=max(self.timeout_s, 600.0),
                )

        uploaded = send_with_retry(send_upload, sleep=self._sleep)
        if uploaded.status_code >= 400:
            raise error_from_response(uploaded, context=f"omi: uploading audio for job {job_id} failed")

        completed = send_with_retry(
            lambda: client.post(complete_url, headers=headers, timeout=self.timeout_s), sleep=self._sleep
        )
        if completed.status_code >= 400:
            raise error_from_response(completed, context=f"omi: completing the upload for job {job_id} failed")

        deadline = time.monotonic() + self.deadline_s
        result = self._poll(client, poll_url, headers, job_id=job_id, deadline=deadline)
        raw = {
            "endpoint": jobs_url,
            "job_id": job_id,
            "parameters": parameters,
            "upload": {"format": self.upload, "content_type": content_type, "bytes": size},
        }
        return self._hypothesis(result, job_id=job_id, language=lang, raw=raw)

    def _poll(self, client: httpx.Client, poll_url: str, headers: dict[str, str], *, job_id: str, deadline: float):
        seen: str | None = None
        while True:
            if time.monotonic() > deadline:
                raise ProviderError(f"omi: job {job_id} did not finish within {int(self.deadline_s)} s", retryable=True)
            started = time.monotonic()
            response = send_with_retry(
                lambda: client.get(
                    poll_url,
                    headers=headers,
                    params={"wait": int(self.poll_wait_s), "include_result": "true"},
                    timeout=self.poll_wait_s + self.timeout_s,
                ),
                sleep=self._sleep,
            )
            long_polled = time.monotonic() - started >= 1.0
            if response.status_code >= 400:
                raise error_from_response(response, context=f"omi: polling job {job_id} failed")
            state = self._json_object(response, context=f"omi: poll of job {job_id}")
            status = state.get("status") or state.get("state")
            if not isinstance(status, str):
                raise ProviderError(f"omi: poll of job {job_id} has no status")
            status = status.lower()
            if status != seen:
                self._status(job_id, status)
                seen = status
            if status in FAILED:
                raise ProviderError(f"omi: job {job_id} {status}: {self._error_message(state)}")
            if status in SUCCEEDED:
                result = self._fetch_result(client, state.get("result"), job_id=job_id, deadline=deadline)
                decoration = result.get("decoration")
                if isinstance(decoration, dict) and decoration.get("status") == "processing":
                    self._status(job_id, "decorating")
                    self._sleep(self.poll_wait_s)
                    continue
                return result
            if status not in PENDING:
                self._status(job_id, f"unknown state {status!r}, still waiting")
            if not long_polled:
                # The server answered immediately instead of holding the request for ``wait`` seconds.
                self._sleep(min(self.poll_wait_s, 5.0))

    def _fetch_result(self, client: httpx.Client, envelope: object, *, job_id: str, deadline: float) -> dict:
        if not isinstance(envelope, dict):
            raise ProviderError(f"omi: job {job_id} succeeded but the poll response has no result")
        content = envelope.get("content")
        if isinstance(content, dict):
            return content
        if isinstance(content, str) and content.strip():
            try:
                parsed = json.loads(content)
            except ValueError as exc:
                raise ProviderError(f"omi: job {job_id} inline result is not valid JSON") from exc
            if not isinstance(parsed, dict):
                raise ProviderError(f"omi: job {job_id} inline result is not a JSON object")
            return parsed
        download_url = envelope.get("download_url")
        if not isinstance(download_url, str) or not download_url:
            raise ProviderError(f"omi: job {job_id} succeeded without result content or a download URL")
        download_url = self._absolute(download_url)
        download_headers = self._auth_for_download(download_url)
        while True:
            if time.monotonic() > deadline:
                raise ProviderError(f"omi: job {job_id} result was still processing at the deadline", retryable=True)
            response = send_with_retry(
                lambda: client.get(download_url, headers=download_headers, timeout=self.timeout_s), sleep=self._sleep
            )
            if response.status_code >= 400:
                raise error_from_response(response, context=f"omi: downloading the result of job {job_id} failed")
            result = self._json_object(response, context=f"omi: result of job {job_id}")
            decoration = result.get("decoration")
            if isinstance(decoration, dict) and decoration.get("status") == "processing":
                self._status(job_id, "decorating")
                self._sleep(self.poll_wait_s)
                continue
            return result

    @staticmethod
    def _json_object(response: httpx.Response, *, context: str) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError(f"{context} returned a non-JSON body: {response.text[:300]}") from exc
        if not isinstance(data, dict):
            raise ProviderError(f"{context} returned a JSON {type(data).__name__}, expected an object")
        return data

    @staticmethod
    def _error_message(state: dict[str, Any]) -> str:
        error = state.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error.get("code") or error)
        if isinstance(error, str) and error:
            return error
        return str(state.get("message") or "unknown error")

    @staticmethod
    def _hypothesis(result: dict[str, Any], *, job_id: str, language: str, raw: dict) -> Hypothesis:
        segments = coerce_segments(result.get("segments"))
        text = result.get("text")
        if not isinstance(text, str):
            if segments:
                text = " ".join(segment["text"] for segment in segments if segment["text"])
            else:
                raise ProviderError(f"omi: job {job_id} result lacks `text` and `segments`")
        duration = result.get("duration")
        detected = result.get("language")
        return Hypothesis(
            text=text.strip(),
            language=detected if isinstance(detected, str) and detected else language,
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


__all__ = ["OmiProvider"]
