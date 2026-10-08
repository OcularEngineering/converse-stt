"""Run a model over a dataset and score the result.

A run directory holds everything needed to audit a score:

``run.json``
    Which model config, dataset and normalization profile produced the run.
``hypotheses.jsonl``
    One line per transcribed track: the raw text the model returned plus
    timing and provenance. Appended as the run progresses; safe to resume.
``failures.jsonl``
    One line per failed attempt with a redacted error message.
``scores.jsonl``
    Track and conversation rows with WER / CER edit counts and the
    normalized texts that were compared.
``summary.json``
    Corpus totals for the leaderboard.
"""

from __future__ import annotations

import fcntl
import json
import os
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ocular_evals import __version__
from ocular_evals.config import ModelConfig
from ocular_evals.dataset import Conversation, Dataset, Track, load_dataset, read_jsonl, sha256_file
from ocular_evals.env import redact
from ocular_evals.metrics import EditCounts, percent, primary_metric, score_text
from ocular_evals.providers import Provider, ProviderError, create_provider
from ocular_evals.text import DEFAULT_PROFILE, PROFILES

RUN_FILE = "run.json"
HYPOTHESES_FILE = "hypotheses.jsonl"
FAILURES_FILE = "failures.jsonl"
SCORES_FILE = "scores.jsonl"
SUMMARY_FILE = "summary.json"
LOCK_FILE = ".lock"
RUN_SCHEMA = 1

Logger = Callable[[str], None]


class RunError(RuntimeError):
    pass


def _log(message: str) -> None:
    print(redact(message), file=sys.stderr, flush=True)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class RunOptions:
    profile: str = DEFAULT_PROFILE
    resume: bool = True
    conversation_ids: tuple[str, ...] = ()
    track_ids: tuple[str, ...] = ()
    max_consecutive_failures: int = 3
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.profile not in PROFILES:
            raise RunError(f"unknown normalization profile {self.profile!r}; expected one of {', '.join(PROFILES)}")


def select_tracks(dataset: Dataset, options: RunOptions) -> list[tuple[Conversation, Track]]:
    wanted_conversations = set(options.conversation_ids)
    wanted_tracks = set(options.track_ids)
    unknown = wanted_conversations - {conversation.id for conversation in dataset.conversations}
    if unknown:
        raise RunError(f"unknown conversation id(s): {', '.join(sorted(unknown))}")
    selected = []
    for conversation, track in dataset.tracks:
        if wanted_conversations and conversation.id not in wanted_conversations:
            continue
        if wanted_tracks and track.id not in wanted_tracks:
            continue
        selected.append((conversation, track))
    if not selected:
        raise RunError("the selection matched no tracks")
    return selected


def _run_record(dataset: Dataset, config: ModelConfig, options: RunOptions) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA,
        "created_at": _now(),
        "harness": {"name": "ocular-evals", "version": __version__},
        "model": config.to_dict(),
        "model_digest": config.digest(),
        "dataset": {"root": str(dataset.root.resolve()), "digest": dataset.digest(), "source": dataset.source},
        "profile": options.profile,
    }


@contextmanager
def run_lock(output: Path):
    """Hold an exclusive lock on ``output`` so two runs never append to the same hypotheses file.

    The lock is advisory (``flock``) and released automatically when the process exits, so a
    crashed run never leaves a stale lock behind.
    """
    output.mkdir(parents=True, exist_ok=True)
    path = output / LOCK_FILE
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            holder = ""
            try:
                holder = os.read(fd, 64).decode("ascii", "replace").strip()
            except OSError:
                pass
            who = f" (pid {holder})" if holder else ""
            raise RunError(
                f"another ocular-evals run is already writing to {output}{who}; wait for it or use --output"
            ) from None
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode("ascii"))
        yield
    finally:
        os.close(fd)


def _check_existing(output: Path, record: dict[str, Any], options: RunOptions) -> dict[str, Any]:
    run_path = output / RUN_FILE
    if not run_path.exists():
        return record
    existing = json.loads(run_path.read_text(encoding="utf-8"))
    if not options.resume:
        return record
    mismatches = []
    if existing.get("model_digest") != record["model_digest"]:
        mismatches.append("model config")
    if existing.get("dataset", {}).get("digest") != record["dataset"]["digest"]:
        mismatches.append("dataset")
    if mismatches:
        raise RunError(
            f"{output} already holds a run for a different {' and '.join(mismatches)}; "
            "use another --output or --fresh to start over"
        )
    existing["profile"] = options.profile
    existing["resumed_at"] = _now()
    return existing


def _completed(output: Path, record: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    path = output / HYPOTHESES_FILE
    if not path.exists():
        return {}
    done: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("model_digest") != record["model_digest"]:
            continue
        done[(row["conversation"], row["track"])] = row
    return done


def transcribe_dataset(
    dataset: Dataset,
    config: ModelConfig,
    output: Path,
    *,
    options: RunOptions | None = None,
    provider: Provider | None = None,
    log: Logger = _log,
) -> dict[str, Any]:
    """Transcribe every selected track, resuming where a previous run stopped.

    Returns a dictionary with the counts of transcribed, skipped and failed tracks.
    Raises :class:`RunError` if another run already holds ``output``.
    """
    options = options or RunOptions()
    with run_lock(output):
        return _transcribe_locked(dataset, config, output, options=options, provider=provider, log=log)


def _transcribe_locked(
    dataset: Dataset,
    config: ModelConfig,
    output: Path,
    *,
    options: RunOptions,
    provider: Provider | None,
    log: Logger,
) -> dict[str, Any]:
    record = _check_existing(output, _run_record(dataset, config, options), options)
    if not options.resume:
        for name in (HYPOTHESES_FILE, FAILURES_FILE, SCORES_FILE, SUMMARY_FILE):
            (output / name).unlink(missing_ok=True)
    _write_json(output / RUN_FILE, record)

    own_provider = provider is None
    provider = provider or create_provider(config)
    problems = provider.check()
    if problems:
        raise RunError(f"{config.name} is not ready to run:\n  - " + "\n  - ".join(problems))

    selection = select_tracks(dataset, options)
    done = _completed(output, record)
    counts = {"transcribed": 0, "skipped": 0, "failed": 0, "selected": len(selection)}
    consecutive_failures = 0
    try:
        for index, (conversation, track) in enumerate(selection, start=1):
            key = (conversation.id, track.id)
            label = f"[{index}/{len(selection)}] {conversation.id}/{track.id}"
            previous = done.get(key)
            if previous and previous.get("audio_sha256") == track.sha256:
                counts["skipped"] += 1
                log(f"{label} already transcribed, skipping")
                continue
            audio = track.audio_path(dataset.root)
            log(f"{label} transcribing {audio.name} ({track.duration_s:.0f}s) with {config.name}")
            started = time.perf_counter()
            try:
                actual = sha256_file(audio) if audio.is_file() else None
                if actual != track.sha256:
                    raise RunError(
                        f"{audio} does not match the manifest (sha256 {actual or 'missing'} != {track.sha256}); "
                        "the dataset is damaged or stale. Run `ocular-evals dataset verify` and `dataset pull`."
                    )
                hypothesis = provider.transcribe(audio, language=conversation.language)
            except RunError as exc:
                failure = {"retryable": False, "status": None, "error": redact(str(exc))}
            except ProviderError as exc:
                failure = {"retryable": exc.retryable, "status": exc.status, "error": redact(str(exc))}
            except Exception as exc:  # noqa: BLE001 - we want every failure on disk
                failure = {"retryable": False, "status": None, "error": redact(f"{type(exc).__name__}: {exc}")}
            else:
                failure = None
                latency = time.perf_counter() - started
                _append_jsonl(
                    output / HYPOTHESES_FILE,
                    {
                        "conversation": conversation.id,
                        "track": track.id,
                        "model": config.name,
                        "model_digest": record["model_digest"],
                        "audio_sha256": track.sha256,
                        "recorded_at": _now(),
                        "latency_s": round(latency, 3),
                        "text": hypothesis.text,
                        "language": hypothesis.language,
                        "duration_s": hypothesis.duration_s,
                        "segments": list(hypothesis.segments),
                        "raw": hypothesis.raw,
                    },
                )
                counts["transcribed"] += 1
                consecutive_failures = 0
                log(f"{label} done in {latency:.1f}s, {len(hypothesis.text.split())} words")
            if failure is not None:
                counts["failed"] += 1
                consecutive_failures += 1
                _append_jsonl(
                    output / FAILURES_FILE,
                    {"conversation": conversation.id, "track": track.id, "recorded_at": _now(), **failure},
                )
                log(f"{label} failed: {failure['error']}")
                if consecutive_failures >= options.max_consecutive_failures:
                    log(f"stopping after {consecutive_failures} consecutive failures")
                    break
    finally:
        if own_provider:
            provider.close()
    return counts


def _latest_hypotheses(output: Path, model_digest: str | None) -> dict[tuple[str, str], dict[str, Any]]:
    path = output / HYPOTHESES_FILE
    if not path.exists():
        raise RunError(f"{path} does not exist; run the model first")
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        if model_digest and row.get("model_digest") not in (None, model_digest):
            continue
        latest[(row["conversation"], row["track"])] = row
    return latest


def score_run(
    output: Path,
    *,
    dataset: Dataset | None = None,
    profile: str | None = None,
) -> dict[str, Any]:
    """Score the hypotheses in ``output`` against the dataset references and write scores + summary."""
    run_path = output / RUN_FILE
    if not run_path.exists():
        raise RunError(f"{run_path} does not exist")
    record = json.loads(run_path.read_text(encoding="utf-8"))
    profile = profile or record.get("profile") or DEFAULT_PROFILE
    if profile not in PROFILES:
        raise RunError(f"unknown normalization profile {profile!r}")
    dataset = dataset or load_dataset(Path(record["dataset"]["root"]))
    if dataset.digest() != record["dataset"]["digest"]:
        raise RunError("the dataset no longer matches the one this run was transcribed from")
    hypotheses = _latest_hypotheses(output, record.get("model_digest"))
    failures = read_jsonl(output / FAILURES_FILE) if (output / FAILURES_FILE).exists() else []

    scores_path = output / SCORES_FILE
    scores_path.unlink(missing_ok=True)
    conversation_rows = []
    corpus_wer = EditCounts()
    corpus_cer = EditCounts()
    scored_tracks = 0
    missing_tracks = 0
    for conversation in dataset.conversations:
        conversation_wer = EditCounts()
        conversation_cer = EditCounts()
        track_rows = []
        for track in conversation.tracks:
            hypothesis = hypotheses.get((conversation.id, track.id))
            if hypothesis is None:
                missing_tracks += 1
                continue
            score = score_text(track.reference, hypothesis["text"], language=conversation.language, profile=profile)
            row = {
                "level": "track",
                "conversation": conversation.id,
                "track": track.id,
                "speaker_id": track.speaker_id,
                "language": conversation.language,
                "profile": profile,
                "offset_s": track.offset_s,
                "duration_s": track.duration_s,
                "latency_s": hypothesis.get("latency_s"),
                "wer": score.wer.to_dict(),
                "cer": score.cer.to_dict(),
                "reference": score.reference,
                "hypothesis": score.hypothesis,
            }
            _append_jsonl(scores_path, row)
            track_rows.append(row)
            conversation_wer += score.wer
            conversation_cer += score.cer
            scored_tracks += 1
        if not track_rows:
            continue
        conversation_row = {
            "level": "conversation",
            "conversation": conversation.id,
            "language": conversation.language,
            "profile": profile,
            "tracks": len(track_rows),
            "complete": len(track_rows) == len(conversation.tracks),
            "wer": conversation_wer.to_dict(),
            "cer": conversation_cer.to_dict(),
        }
        _append_jsonl(scores_path, conversation_row)
        conversation_rows.append(
            {
                **conversation_row,
                "track_scores": [
                    {key: value for key, value in row.items() if key not in {"reference", "hypothesis", "level"}}
                    for row in track_rows
                ],
            }
        )
        corpus_wer += conversation_wer
        corpus_cer += conversation_cer

    languages = {conversation.language for conversation in dataset.conversations}
    primary = primary_metric(next(iter(languages))) if len(languages) == 1 else "wer"
    summary = {
        "schema": RUN_SCHEMA,
        "scored_at": _now(),
        "model": record["model"],
        "model_digest": record.get("model_digest"),
        "dataset": record["dataset"],
        "profile": profile,
        "primary_metric": primary,
        "wer": corpus_wer.to_dict(),
        "cer": corpus_cer.to_dict(),
        "tracks": {"scored": scored_tracks, "missing": missing_tracks, "total": len(dataset.tracks)},
        "failures": len(failures),
        "conversations": conversation_rows,
    }
    _write_json(output / SUMMARY_FILE, summary)
    return summary


def format_summary(summary: dict[str, Any]) -> str:
    """Human readable summary for the terminal."""
    model = summary["model"]
    lines = [
        f"{model.get('display_name') or model['name']} ({model['provider']}:{model['model']})",
        f"  profile   {summary['profile']}",
        f"  WER       {percent(summary['wer']['rate'])}  "
        f"({summary['wer']['errors']:,} errors / {summary['wer']['reference_length']:,} words)",
        f"  CER       {percent(summary['cer']['rate'])}  "
        f"({summary['cer']['errors']:,} errors / {summary['cer']['reference_length']:,} chars)",
        f"  tracks    {summary['tracks']['scored']}/{summary['tracks']['total']} scored, "
        f"{summary['failures']} failed attempt(s)",
    ]
    for conversation in summary["conversations"]:
        flag = "" if conversation["complete"] else "  (incomplete)"
        lines.append(
            f"  {conversation['conversation']}  WER {percent(conversation['wer']['rate'])}  "
            f"CER {percent(conversation['cer']['rate'])}{flag}"
        )
    return "\n".join(lines)


def load_summary(output: Path) -> dict[str, Any]:
    path = output / SUMMARY_FILE
    if not path.exists():
        raise RunError(f"{path} does not exist; run `ocular-evals score --run {output}` first")
    return json.loads(path.read_text(encoding="utf-8"))
