"""Build the Converse-STT 2.0 review page from one or more scored runs.

The page is a single self-contained HTML file. It ranks the harness runs
(optionally next to the Converse-STT 1.0 board) and lets you open every conversation to read
the normalized reference beside the model's transcript.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

from ocular_evals.dataset import read_jsonl
from ocular_evals.runner import HYPOTHESES_FILE, SCORES_FILE, SUMMARY_FILE, RunError, load_summary

DEFAULT_BOARD = "converse-stt"
EXCERPT_CHARS = 420
REPEAT_THRESHOLD = 8


def empty_board() -> dict[str, Any]:
    """The default: no published rows, only harness runs."""
    return {"name": "Converse-STT 2.0", "label": "", "updated": "", "source": "", "metric": "wer", "models": []}


def discover_runs(runs_dir: Path) -> list[Path]:
    """Every direct child of ``runs_dir`` that has been scored, oldest first."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    found = [child for child in runs_dir.iterdir() if child.is_dir() and (child / SUMMARY_FILE).is_file()]
    return sorted(found, key=lambda child: ((child / SUMMARY_FILE).stat().st_mtime, child.name))


def load_board(name_or_path: str | Path | None) -> dict[str, Any]:
    """Load a published board (built-in name such as ``converse-stt`` or a JSON path); ``None`` means no board."""
    if name_or_path is None:
        return empty_board()
    path = Path(name_or_path)
    if path.suffix == ".json" and path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        resource = resources.files("ocular_evals").joinpath("boards", f"{name_or_path}.json")
        if not resource.is_file():
            raise RunError(f"unknown board {name_or_path!r}")
        data = json.loads(resource.read_text(encoding="utf-8"))
    models = data.get("models")
    if not isinstance(models, list):
        raise RunError("a board file needs a `models` list")
    data["models"] = [
        {**model, "rank": index, "kind": "published"} for index, model in enumerate(models, start=1) if "wer" in model
    ]
    return data


def _excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rsplit(" ", 1)[0] + "…"


def _repeated(segments: list[dict[str, Any]]) -> dict[str, Any] | None:
    texts = [str(segment.get("text") or "").strip() for segment in segments]
    texts = [text for text in texts if text]
    if not texts:
        return None
    text, count = Counter(texts).most_common(1)[0]
    if count < REPEAT_THRESHOLD:
        return None
    return {"text": text, "count": count, "segments": len(texts)}


def _pct(rate: float | None) -> float | None:
    return None if rate is None else round(rate * 100, 2)


def _title(reference: str, fallback: str) -> str:
    words = reference.split()[:8]
    title = " ".join(words).rstrip(".,")
    if not title:
        return fallback
    return title[:1].upper() + title[1:]


def run_payload(run_dir: Path) -> dict[str, Any]:
    """Collect everything the page needs about one run directory."""
    summary = load_summary(run_dir)
    tracks = {
        (row["conversation"], row["track"]): row
        for row in read_jsonl(run_dir / SCORES_FILE)
        if row.get("level") == "track"
    }
    hypotheses = {(row["conversation"], row["track"]): row for row in read_jsonl(run_dir / HYPOTHESES_FILE)}
    conversations = []
    for index, conversation in enumerate(summary["conversations"], start=1):
        channels = []
        for track_score in conversation["track_scores"]:
            key = (conversation["conversation"], track_score["track"])
            row = tracks[key]
            hypothesis = hypotheses.get(key, {})
            channels.append(
                {
                    "channel": row["track"],
                    "speakerId": row.get("speaker_id", ""),
                    "offsetS": row.get("offset_s", 0.0),
                    "durationS": row.get("duration_s"),
                    "latencyS": round(float(row.get("latency_s") or 0), 1),
                    "wer": _pct(row["wer"]["rate"]),
                    "cer": _pct(row["cer"]["rate"]),
                    "hits": row["wer"]["hits"],
                    "substitutions": row["wer"]["substitutions"],
                    "deletions": row["wer"]["deletions"],
                    "insertions": row["wer"]["insertions"],
                    "denominator": row["wer"]["reference_length"],
                    "cerErrors": row["cer"]["errors"],
                    "cerDenominator": row["cer"]["reference_length"],
                    "reference": _excerpt(row["reference"]),
                    "hypothesis": _excerpt(row["hypothesis"]),
                    "repeated": _repeated(hypothesis.get("segments") or []),
                }
            )
        first = tracks.get((conversation["conversation"], channels[0]["channel"])) if channels else None
        conversations.append(
            {
                "id": conversation["conversation"],
                "number": f"{index:02d}",
                "title": _title(first["reference"] if first else "", f"Conversation {index}"),
                "complete": conversation["complete"],
                "wer": _pct(conversation["wer"]["rate"]),
                "cer": _pct(conversation["cer"]["rate"]),
                "channels": channels,
            }
        )
    model = summary["model"]
    return {
        "id": model["name"],
        "name": model.get("display_name") or model["name"],
        "vendor": model.get("vendor") or model["provider"],
        "modelId": model["model"],
        "href": model.get("href") or "",
        "profile": summary["profile"],
        "wer": _pct(summary["wer"]["rate"]),
        "cer": _pct(summary["cer"]["rate"]),
        "numerator": summary["wer"]["errors"],
        "denominator": summary["wer"]["reference_length"],
        "cerNumerator": summary["cer"]["errors"],
        "cerDenominator": summary["cer"]["reference_length"],
        "tracks": summary["tracks"]["scored"],
        "tracksTotal": summary["tracks"]["total"],
        "failures": summary["failures"],
        "conversations": conversations,
    }


def build_payload(
    run_dirs: list[Path],
    *,
    board: dict[str, Any] | None = None,
    sample_label: str = "This sample",
) -> dict[str, Any]:
    runs = [run_payload(Path(run_dir)) for run_dir in run_dirs]
    runs = [run for run in runs if run["wer"] is not None]
    seen: set[str] = set()
    for run in runs:
        base = run["id"]
        suffix = 2
        while run["id"] in seen:
            run["id"] = f"{base}-{suffix}"
            suffix += 1
        seen.add(run["id"])
    board = board if board is not None else empty_board()
    return {
        "generated": datetime.now(UTC).strftime("%b %-d, %Y"),
        "sampleLabel": sample_label,
        "board": {key: value for key, value in board.items() if key != "models"},
        "published": board["models"],
        "runs": runs,
    }


def render_html(payload: dict[str, Any]) -> str:
    template = resources.files("ocular_evals").joinpath("templates", "leaderboard.html").read_text(encoding="utf-8")
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    return template.replace("/*__DATA__*/ null", data)


def write_report(
    run_dirs: list[Path],
    output: Path,
    *,
    board: str | Path | None = None,
    sample_label: str = "This sample",
) -> Path:
    payload = build_payload(run_dirs, board=load_board(board), sample_label=sample_label)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(payload), encoding="utf-8")
    return output
