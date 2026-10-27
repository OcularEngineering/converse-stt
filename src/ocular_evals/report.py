"""Build the Converse-STT 2.0 review page from one or more scored runs.

The page is a single self-contained HTML file. It ranks the harness runs
(optionally next to the Converse-STT 1.0 board) and lets you open every conversation to read
the normalized reference beside the model's transcript.
"""

from __future__ import annotations

import base64
import json
from collections import Counter
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

from ocular_evals.dataset import read_jsonl
from ocular_evals.metrics import alignment_ops
from ocular_evals.runner import HYPOTHESES_FILE, SCORES_FILE, SUMMARY_FILE, RunError, load_summary

DEFAULT_BOARD = "converse-stt"
REPEAT_THRESHOLD = 8
LOCAL_PROVIDERS = frozenset({"faster-whisper", "nemotron-asr", "cohere-transcribe"})


# Vendor marks shipped with the package (``templates/logos``), keyed by the vendor name in models.toml.
VENDOR_LOGOS = {
    "Alibaba": "qwen.png",
    "Alibaba Qwen": "qwen.png",
    "Qwen": "qwen.png",
    "Amazon": "amazon.svg",
    "AssemblyAI": "assemblyai.png",
    "Cartesia": "cartesia.svg",
    "Cohere": "cohere.png",
    "Deepgram": "deepgram.svg",
    "ElevenLabs": "elevenlabs.svg",
    "Google": "google.png",
    "Gradium": "gradium.svg",
    "Inworld": "inworld.png",
    "Meta": "meta.svg",
    "Microsoft": "microsoft.svg",
    "Microsoft AI": "microsoft.svg",
    "MiniMax": "minimax.svg",
    "Mistral": "mistral.svg",
    "Mistral AI": "mistral.svg",
    "NVIDIA": "nvidia.svg",
    "Ocular": "ocular.svg",
    "Omi": "omi.png",
    "Omi Health": "omi.png",
    "OpenAI": "openai.svg",
    "Reson8": "reson8.png",
    "Smallest": "smallest.png",
    "Speechmatics": "speechmatics.png",
    "xAI": "xai.png",
}
# White versions of marks that are near-black on transparent, for the website's dark mode. A vendor
# listed here is exported as a ``{"light", "dark"}`` pair; everyone else as a single path.
VENDOR_LOGOS_DARK = {
    "ElevenLabs": "elevenlabs-dark.svg",
    "Mistral": "mistral-dark.svg",
    "Mistral AI": "mistral-dark.svg",
}
_MIME = {".svg": "image/svg+xml", ".png": "image/png"}


def vendor_logo(vendor: str) -> str | None:
    """The vendor's mark as a ``data:`` URL, or ``None`` when the package has no mark for it."""
    name = VENDOR_LOGOS.get(vendor)
    if not name:
        return None
    resource = resources.files("ocular_evals").joinpath("templates", "logos", name)
    if not resource.is_file():
        return None
    mime = _MIME[Path(name).suffix]
    return f"data:{mime};base64,{base64.b64encode(resource.read_bytes()).decode('ascii')}"


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


def _median(values: list[float]) -> float | None:
    values = sorted(values)
    if not values:
        return None
    middle = len(values) // 2
    if len(values) % 2:
        return round(values[middle], 1)
    return round((values[middle - 1] + values[middle]) / 2, 1)


def _route(model: dict[str, Any]) -> str:
    if model.get("provider") in LOCAL_PROVIDERS:
        return "local"
    base_url = str((model.get("options") or {}).get("base_url") or "")
    return "openrouter" if "openrouter" in base_url else "direct"


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
                    "reference": row["reference"],
                    "hypothesis": row["hypothesis"],
                    "raw": str(hypothesis.get("text") or ""),
                    "ops": [list(op) for op in alignment_ops(row["reference"].split(), row["hypothesis"].split())],
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
                "durationS": round(max([(c["durationS"] or 0) + (c["offsetS"] or 0) for c in channels] or [0]), 3),
                "wer": _pct(conversation["wer"]["rate"]),
                "cer": _pct(conversation["cer"]["rate"]),
                "substitutions": conversation["wer"]["substitutions"],
                "deletions": conversation["wer"]["deletions"],
                "insertions": conversation["wer"]["insertions"],
                "channels": channels,
            }
        )
    model = summary["model"]
    return {
        "id": model["name"],
        "name_key": model["name"],
        "name": model.get("display_name") or model["name"],
        "vendor": model.get("vendor") or model["provider"],
        "modelId": model["model"],
        "href": model.get("href") or "",
        "notes": model.get("notes") or "",
        "group": model.get("group") or "",
        "provider": model.get("provider") or "",
        "route": _route(model),
        "options": {k: v for k, v in (model.get("options") or {}).items() if not str(k).endswith("_env")},
        "profile": summary["profile"],
        "scoredAt": summary.get("scored_at"),
        "wer": _pct(summary["wer"]["rate"]),
        "cer": _pct(summary["cer"]["rate"]),
        "numerator": summary["wer"]["errors"],
        "denominator": summary["wer"]["reference_length"],
        "substitutions": summary["wer"]["substitutions"],
        "deletions": summary["wer"]["deletions"],
        "insertions": summary["wer"]["insertions"],
        "cerNumerator": summary["cer"]["errors"],
        "cerDenominator": summary["cer"]["reference_length"],
        "cerSubstitutions": summary["cer"]["substitutions"],
        "cerDeletions": summary["cer"]["deletions"],
        "cerInsertions": summary["cer"]["insertions"],
        "latencyS": _median(
            [float(c["latencyS"]) for item in conversations for c in item["channels"] if c["latencyS"]]
        ),
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
    everything = [run_payload(Path(run_dir)) for run_dir in run_dirs]
    everything = [run for run in everything if run["wer"] is not None]
    seen: set[str] = set()
    for run in everything:
        base = run["id"]
        suffix = 2
        while run["id"] in seen:
            run["id"] = f"{base}-{suffix}"
            suffix += 1
        seen.add(run["id"])
    board = board if board is not None else empty_board()
    board_group = board.get("group")
    open_weights = set(board.get("openWeights") or [])
    streaming = set(board.get("streaming") or [])
    for run in everything:
        hosted_weights = run["href"].startswith("https://huggingface.co/")
        is_open = run["name_key"] in open_weights or run["route"] == "local" or hosted_weights
        run["access"] = "open" if is_open else "proprietary"
        run["streaming"] = run["name_key"] in streaming
    # Conversation topics and speakers come from the board file when it describes this dataset;
    # otherwise the first words of the reference stand in for a title.
    described = board.get("conversations") or {}
    for run in everything:
        for item in run["conversations"]:
            meta = described.get(item["id"].replace("ocular-", "")[:8]) or described.get(item["id"])
            if meta:
                item["topic"] = meta.get("topic") or item["title"]
                item["speakers"] = meta.get("speakers") or {}
            else:
                item["topic"] = item["title"]
                item["speakers"] = {}
    # Runs tagged with a group (``group = "converse-stt-1.0"`` in models.toml) belong to that tab, not the 2.0 board.
    runs = [run for run in everything if not run["group"]]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in everything:
        if run["group"]:
            grouped.setdefault(run["group"], []).append(run)
    by_name = {run["name_key"]: run for run in everything}
    published = []
    matched: set[str] = set()
    for model in board["models"]:
        entry = dict(model)
        rerun = by_name.get(str(model.get("run") or ""))
        if rerun is not None:
            entry["rerun"] = rerun["id"]
            matched.add(rerun["id"])
        published.append(entry)
    # Group runs that no published row claims are still listed on the group's tab.
    extra = [run for run in grouped.get(board_group or "", []) if run["id"] not in matched]
    sample = runs[0] if runs else (everything[0] if everything else None)
    dataset = {
        "conversations": len(sample["conversations"]) if sample else 0,
        "tracks": sample["tracksTotal"] if sample else 0,
        "referenceWords": sample["denominator"] if sample else 0,
        "referenceCharacters": sample["cerDenominator"] if sample else 0,
    }
    vendors = {run["vendor"] for run in everything} | {str(model.get("vendor") or "") for model in published}
    logos = {vendor: logo for vendor in sorted(vendors) if (logo := vendor_logo(vendor))}
    return {
        "generated": datetime.now(UTC).strftime("%b %-d, %Y"),
        "sampleLabel": sample_label,
        "dataset": dataset,
        "logos": logos,
        "board": {key: value for key, value in board.items() if key != "models"},
        "published": published,
        "runs": runs,
        "groupRuns": grouped,
        "groupExtra": [run["id"] for run in extra],
        "allRuns": everything,
    }


def render_html(payload: dict[str, Any]) -> str:
    template = resources.files("ocular_evals").joinpath("templates", "leaderboard.html").read_text(encoding="utf-8")
    data = json.dumps(_compact(payload), ensure_ascii=False).replace("<", "\\u003c")
    return template.replace("/*__DATA__*/ null", data)


def _compact(payload: dict[str, Any]) -> dict[str, Any]:
    """The payload as embedded in the page: every run once, and each reference text once.

    ``runs`` and ``groupRuns`` become lists of run ids resolved against ``allRuns``, and the
    per-track reference texts (identical for every run) move to a ``references`` map keyed
    ``conversation/track``. The template reverses both.
    """
    all_runs = payload.get("allRuns")
    if all_runs is None:
        return payload
    references: dict[str, str] = {}
    compact_runs = []
    for run in all_runs:
        conversations = []
        for item in run["conversations"]:
            channels = []
            for channel in item["channels"]:
                key = f"{item['id']}/{channel['channel']}"
                references.setdefault(key, channel.get("reference", ""))
                channels.append({k: v for k, v in channel.items() if k != "reference"})
            conversations.append({**item, "channels": channels})
        compact_runs.append({**run, "conversations": conversations})
    return {
        **payload,
        "allRuns": compact_runs,
        "runs": [run["id"] for run in payload["runs"]],
        "groupRuns": {group: [run["id"] for run in runs] for group, runs in payload.get("groupRuns", {}).items()},
        "references": references,
    }


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
