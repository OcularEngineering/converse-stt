"""OpenAI Realtime API transcription sessions over WebSocket (``whisper-1``, ``gpt-4o-transcribe``, ...).

This is the streaming path a voice agent uses: 24 kHz PCM16 audio is appended to the server's input
buffer, server-side voice activity detection cuts it into speech turns, and each turn comes back as a
``conversation.item.input_audio_transcription.completed`` event. The provider streams a whole track at
``speed`` times real time, commits the tail, waits for every committed turn to be transcribed and joins the
turns in order. Converse-STT 1.0 scored "GPT Realtime Whisper" through this API; this adapter reproduces it.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ocular_evals.config import ModelConfig
from ocular_evals.providers.audio import audio_info
from ocular_evals.providers.base import Hypothesis, Provider, ProviderError, primary_language_tag

REALTIME_URL = "wss://api.openai.com/v1/realtime?intent=transcription"
SAMPLE_RATE = 24_000
CHUNK_S = 0.1
DEFAULT_SPEED = 10.0  # audio seconds sent per wall second
DEFAULT_SILENCE_MS = 500
DEFAULT_IDLE_TIMEOUT_S = 120.0
RETRYABLE_CODES = {"rate_limit_exceeded", "server_error", "session_expired"}


def _import_websockets():
    try:
        import websockets
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ProviderError("the websockets package is required for the openai-realtime provider") from exc
    return websockets


def _load_pcm24k(path: Path) -> bytes:
    try:
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:
        raise ProviderError(f"cannot decode {path.name}: {exc}") from exc
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if rate != SAMPLE_RATE and mono.size:
        count = int(round(mono.size / float(rate) * SAMPLE_RATE))
        src_t = np.arange(mono.size, dtype=np.float64) / float(rate)
        dst_t = np.arange(count, dtype=np.float64) / float(SAMPLE_RATE)
        mono = np.interp(dst_t, src_t, mono.astype(np.float64)).astype(np.float32)
    return (np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


class OpenAIRealtimeProvider(Provider):
    key = "openai-realtime"
    option_names = frozenset(
        {
            "api_key_env",
            "speed",
            "silence_ms",
            "vad_threshold",
            "prefix_padding_ms",
            "language_hint",
            "prompt",
            "idle_timeout_s",
        }
    )

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        options = config.options
        self.api_key_env = str(options.get("api_key_env") or (config.env[0] if config.env else "OPENAI_API_KEY"))
        self.speed = options.get("speed", DEFAULT_SPEED)
        self.silence_ms = options.get("silence_ms", DEFAULT_SILENCE_MS)
        self.vad_threshold = options.get("vad_threshold")
        self.prefix_padding_ms = options.get("prefix_padding_ms")
        self.language_hint = options.get("language_hint", True)
        self.prompt = options.get("prompt")
        self.idle_timeout_s = options.get("idle_timeout_s", DEFAULT_IDLE_TIMEOUT_S)

    def required_env(self) -> tuple[str, ...]:
        return (self.api_key_env,)

    def check(self) -> list[str]:
        problems = super().check()
        for name in ("speed", "silence_ms", "idle_timeout_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                problems.append(f"{name} must be a positive number")
        if self.vad_threshold is not None and not (
            isinstance(self.vad_threshold, (int, float))
            and not isinstance(self.vad_threshold, bool)
            and 0 <= self.vad_threshold <= 1
        ):
            problems.append("vad_threshold must be between 0 and 1")
        if self.prefix_padding_ms is not None and (
            isinstance(self.prefix_padding_ms, bool)
            or not isinstance(self.prefix_padding_ms, int)
            or self.prefix_padding_ms < 0
        ):
            problems.append("prefix_padding_ms must be a non-negative integer")
        if not isinstance(self.language_hint, bool):
            problems.append("language_hint must be true or false")
        if self.prompt is not None and not isinstance(self.prompt, str):
            problems.append("prompt must be a string")
        return problems

    def _session_update(self, language: str) -> dict[str, Any]:
        transcription: dict[str, Any] = {"model": self.config.model}
        if self.language_hint:
            transcription["language"] = primary_language_tag(language)
        if self.prompt:
            transcription["prompt"] = self.prompt
        turn_detection: dict[str, Any] = {"type": "server_vad", "silence_duration_ms": int(self.silence_ms)}
        if self.vad_threshold is not None:
            turn_detection["threshold"] = float(self.vad_threshold)
        if self.prefix_padding_ms is not None:
            turn_detection["prefix_padding_ms"] = int(self.prefix_padding_ms)
        return {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                        "transcription": transcription,
                        "turn_detection": turn_detection,
                    }
                },
            },
        }

    async def _stream(self, pcm: bytes, language: str) -> dict[str, Any]:
        websockets = _import_websockets()
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise ProviderError(f"environment variable {self.api_key_env} is not set")
        chunk = int(CHUNK_S * SAMPLE_RATE) * 2
        pause = CHUNK_S / float(self.speed)
        order: list[str] = []
        transcripts: dict[str, str] = {}
        bounds: dict[str, dict[str, Any]] = {}
        events: dict[str, int] = {}
        pending = 0
        sent_all = False
        last_start_ms: int | None = None
        connect = websockets.connect(
            REALTIME_URL,
            additional_headers={"Authorization": f"Bearer {api_key}"},
            max_size=None,
            open_timeout=30,
        )
        async with connect as ws:
            await ws.send(json.dumps(self._session_update(language)))

            async def sender() -> None:
                nonlocal sent_all
                for index in range(0, len(pcm), chunk):
                    payload = base64.b64encode(pcm[index : index + chunk]).decode("ascii")
                    await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": payload}))
                    await asyncio.sleep(pause)
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                sent_all = True

            async def receiver() -> None:
                nonlocal pending, last_start_ms
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=float(self.idle_timeout_s))
                    except TimeoutError:
                        if sent_all and pending <= 0:
                            return
                        raise ProviderError(
                            f"realtime session idle for {self.idle_timeout_s}s with {pending} turn(s) outstanding",
                            retryable=True,
                        ) from None
                    message = json.loads(raw)
                    kind = str(message.get("type") or "")
                    events[kind] = events.get(kind, 0) + 1
                    if kind == "error":
                        error = message.get("error") or {}
                        code = str(error.get("code") or "")
                        if code == "input_audio_buffer_commit_empty":
                            # The tail after the last turn was silence; nothing left to transcribe.
                            if sent_all and pending <= 0:
                                return
                            continue
                        raise ProviderError(
                            f"realtime API error {code or 'unknown'}: {error.get('message', '')}",
                            retryable=code in RETRYABLE_CODES,
                        )
                    if kind == "input_audio_buffer.speech_started":
                        last_start_ms = message.get("audio_start_ms")
                    elif kind == "input_audio_buffer.committed":
                        item_id = str(message.get("item_id"))
                        order.append(item_id)
                        bounds[item_id] = {"start_ms": last_start_ms}
                        pending += 1
                    elif kind == "input_audio_buffer.speech_stopped":
                        if order:
                            bounds.setdefault(order[-1], {})["end_ms"] = message.get("audio_end_ms")
                    elif kind in (
                        "conversation.item.input_audio_transcription.completed",
                        "conversation.item.input_audio_transcription.failed",
                    ):
                        item_id = str(message.get("item_id"))
                        transcripts[item_id] = str(message.get("transcript") or "")
                        if kind.endswith("failed"):
                            events["transcription_failed_items"] = events.get("transcription_failed_items", 0) + 1
                        pending -= 1
                        if sent_all and pending <= 0:
                            return

            sender_task = asyncio.create_task(sender())
            try:
                await receiver()
            finally:
                if not sender_task.done():
                    sender_task.cancel()
            try:
                await ws.close()
            except Exception:
                pass
        return {"order": order, "transcripts": transcripts, "bounds": bounds, "events": events}

    def transcribe(self, audio: Path, *, language: str) -> Hypothesis:
        pcm = _load_pcm24k(audio)
        duration_s, _ = audio_info(audio)
        started = time.monotonic()
        try:
            result = asyncio.run(self._stream(pcm, language))
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"realtime transcription failed on {audio.name}: {exc}", retryable=True) from exc
        segments = []
        texts = []
        for item_id in result["order"]:
            text = result["transcripts"].get(item_id, "").strip()
            bound = result["bounds"].get(item_id, {})
            start = bound.get("start_ms")
            end = bound.get("end_ms")
            segments.append(
                {
                    "start_s": start / 1000.0 if isinstance(start, (int, float)) else None,
                    "end_s": end / 1000.0 if isinstance(end, (int, float)) else None,
                    "text": text,
                }
            )
            if text:
                texts.append(text)
        return Hypothesis(
            text=" ".join(texts),
            language=primary_language_tag(language) if self.language_hint else None,
            duration_s=duration_s,
            segments=tuple(segments),
            raw={
                "model": self.config.model,
                "endpoint": REALTIME_URL,
                "session": self._session_update(language)["session"],
                "speed": float(self.speed),
                "turns": len(result["order"]),
                "events": result["events"],
                "wall_s": round(time.monotonic() - started, 1),
            },
        )


__all__ = ["OpenAIRealtimeProvider"]
