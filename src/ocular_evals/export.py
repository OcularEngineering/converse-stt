"""Export scored runs as the JSON the Converse-STT website consumes.

The website never reads a run directory. ``ocular-evals export`` turns ``runs/`` (the board) and
``runs-extra/`` (operating points: the same model at another setting) into::

    <out>/<slug>/v<version>/board.json                 every run with pooled counts per conversation and track
    <out>/<slug>/v<version>/transcripts/<run-id>.json  reference, normalized and raw hypothesis, word alignment
    <out>/<slug>/v<version>/logos/<file>               the vendor marks the board refers to

``slug`` and ``version`` come from the board file (``boards/converse-stt.json``).
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from importlib import metadata, resources
from pathlib import Path
from typing import Any

from ocular_evals.registry import Registry
from ocular_evals.report import DEFAULT_BOARD, LOCAL_PROVIDERS, VENDOR_LOGOS, _median, _route, run_payload
from ocular_evals.runner import RUN_FILE, RunError, load_summary

HARNESS_NAME = "ocular-evals"
HARNESS_HREF = "https://github.com/OcularEngineering/converse-stt"
DEFAULT_EXTRA = Path("runs-extra")
DEFAULT_EXPORT = Path("export")
BOARD_FILE = "board.json"
TRANSCRIPTS_DIR = "transcripts"
LOGOS_DIR = "logos"
_HUGGINGFACE = "https://huggingface.co/"


def harness_version() -> str:
    try:
        return metadata.version(HARNESS_NAME)
    except metadata.PackageNotFoundError:  # pragma: no cover - only when run from an unpacked checkout
        return "0.1.0"


def default_generated() -> str:
    return datetime.now(UTC).strftime("%b %-d, %Y")


def _pct(rate: float) -> float:
    return round(rate * 100, 2)


def _counts(block: dict[str, Any]) -> dict[str, Any]:
    return {
        "rate": _pct(block["rate"]),
        "hits": block["hits"],
        "substitutions": block["substitutions"],
        "deletions": block["deletions"],
        "insertions": block["insertions"],
        "errors": block["errors"],
        "reference_length": block["reference_length"],
    }


def _short(conversation_id: str) -> str:
    return conversation_id.replace("ocular-", "")[:8]


def _conversation_duration(conversation: dict[str, Any]) -> float:
    return round(max(t["duration_s"] + t["offset_s"] for t in conversation["track_scores"]), 3)


def is_prerelease(summary: dict[str, Any], registry: Registry) -> bool:
    """Pre-release models stay off the export: the run's own config says so, or the current ``models.toml`` does."""
    model = summary["model"]
    if model.get("prerelease"):
        return True
    entry = registry.entries.get(str(model.get("name") or ""), {})
    return bool(entry.get("prerelease"))


def convert_run(
    run_dir: Path,
    *,
    board: dict[str, Any],
    registry: Registry,
    run_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One run directory as a board row and its transcripts file.

    ``run_id`` defaults to the model name; operating points pass their directory name so the same
    model can appear once per setting.
    """
    run_dir = Path(run_dir)
    summary = load_summary(run_dir)
    record = json.loads((run_dir / RUN_FILE).read_text(encoding="utf-8"))
    payload = run_payload(run_dir)  # normalized texts, raw hypotheses and word alignments per track
    channels = {(item["id"], c["channel"]): c for item in payload["conversations"] for c in item["channels"]}
    model = summary["model"]
    name = str(model["name"])
    run_id = run_id or name
    entry = registry.entries.get(name, {})

    conversations, transcripts = [], []
    for conversation in summary["conversations"]:
        tracks, texts = [], []
        for score in conversation["track_scores"]:
            channel = channels[(conversation["conversation"], score["track"])]
            tracks.append(
                {
                    "track": score["track"],
                    "duration_s": round(score["duration_s"], 3),
                    "offset_s": score["offset_s"],
                    "latency_s": score.get("latency_s"),
                    "wer": _counts(score["wer"]),
                    "cer": _counts(score["cer"]),
                }
            )
            texts.append(
                {
                    "track": score["track"],
                    "reference": channel["reference"],
                    "hypothesis": channel["hypothesis"],
                    "raw": channel["raw"],
                    "ops": channel["ops"],
                }
            )
        conversations.append(
            {
                "id": conversation["conversation"],
                "complete": conversation["complete"],
                "duration_s": _conversation_duration(conversation),
                "wer": _pct(conversation["wer"]["rate"]),
                "cer": _pct(conversation["cer"]["rate"]),
                "counts": {"wer": _counts(conversation["wer"]), "cer": _counts(conversation["cer"])},
                "tracks": tracks,
            }
        )
        transcripts.append({"id": conversation["conversation"], "tracks": texts})

    href = model.get("href") or entry.get("href")
    open_weights = set(board.get("openWeights") or [])
    is_open = (
        name in open_weights or model.get("provider") in LOCAL_PROVIDERS or str(href or "").startswith(_HUGGINGFACE)
    )
    latencies = [
        float(t["latency_s"])
        for c in summary["conversations"]
        for t in c["track_scores"]
        if t.get("latency_s") is not None
    ]
    run = {
        "id": run_id,
        "displayName": model.get("display_name") or name,
        "vendor": model.get("vendor") or model.get("provider"),
        "model": model["model"],
        "href": href,
        "provider": model.get("provider"),
        "route": _route(model),
        "access": "open" if is_open else "proprietary",
        "streaming": name in set(board.get("streaming") or []),
        "notes": model.get("notes"),
        "options": payload["options"],
        "wer": _pct(summary["wer"]["rate"]),
        "cer": _pct(summary["cer"]["rate"]),
        "counts": {"wer": _counts(summary["wer"]), "cer": _counts(summary["cer"])},
        "medianLatencyS": _median(latencies),
        "tracksScored": summary["tracks"]["scored"],
        "tracksTotal": summary["tracks"]["total"],
        "failures": summary["failures"],
        "complete": summary["tracks"]["missing"] == 0,
        "scoredAt": summary["scored_at"],
        "modelDigest": summary["model_digest"],
        "harness": record["harness"],
        "conversations": conversations,
    }
    transcript = {
        "run": run_id,
        "model": run["displayName"],
        "profile": summary["profile"],
        "conversations": transcripts,
    }
    return run, transcript


def _conversation_meta(summary: dict[str, Any], board: dict[str, Any], payload_titles: dict[str, str]) -> list:
    described = board.get("conversations") or {}
    out = []
    for conversation in summary["conversations"]:
        conversation_id = conversation["conversation"]
        meta = described.get(_short(conversation_id)) or described.get(conversation_id) or {}
        out.append(
            {
                "id": conversation_id,
                "short": _short(conversation_id),
                "topic": meta.get("topic") or payload_titles.get(conversation_id, conversation_id),
                "speakers": meta.get("speakers") or {},
                "duration_s": _conversation_duration(conversation),
                "reference_words": conversation["wer"]["reference_length"],
                "tracks": [
                    {"track": t["track"], "offset_s": t["offset_s"], "reference_words": t["wer"]["reference_length"]}
                    for t in conversation["track_scores"]
                ],
            }
        )
    return out


def build_export(
    run_dirs: list[Path],
    extra_dirs: list[Path],
    *,
    board: dict[str, Any],
    registry: Registry,
    generated: str | None = None,
    include_prerelease: bool = False,
) -> dict[str, Any]:
    """Everything ``write_export`` writes: ``board``, ``transcripts`` (per board run) and the ``excluded`` names."""
    slug = str(board.get("slug") or DEFAULT_BOARD)
    version = str(board.get("version") or "2")
    data_path = f"/data/{slug}/v{version}"

    excluded: list[str] = []
    kept: list[Path] = []
    kept_extra: list[Path] = []
    for dirs, target in ((run_dirs, kept), (extra_dirs, kept_extra)):
        for run_dir in dirs:
            run_dir = Path(run_dir)
            if not include_prerelease and is_prerelease(load_summary(run_dir), registry):
                excluded.append(run_dir.name)
                continue
            target.append(run_dir)
    if not kept:
        raise RunError("no runs to export" + (f" ({len(excluded)} pre-release run(s) excluded)" if excluded else ""))

    converted = [(run_dir, *convert_run(run_dir, board=board, registry=registry)) for run_dir in kept]
    converted.sort(key=lambda item: (item[1]["wer"], item[1]["id"]))  # best WER first; ties by run id
    runs = [run for _, run, _ in converted]
    transcripts = [transcript for _, _, transcript in converted]
    extra = [
        convert_run(run_dir, board=board, registry=registry, run_id=run_dir.name)[0]
        for run_dir in sorted(kept_extra, key=lambda path: path.name)
    ]

    first_dir = converted[0][0]  # the dataset block describes the sample the best run was scored on
    first = load_summary(first_dir)
    titles = {item["id"]: item["title"] for item in run_payload(first_dir)["conversations"]}
    source = first["dataset"].get("source") or {}
    # Marks for every vendor the site can show: board runs, operating points, and the published
    # Version 1 rows (the site keeps a page for them).
    vendors = {str(run["vendor"]) for run in runs + extra}
    vendors |= {str(model.get("vendor") or "") for model in board.get("models", [])}
    vendors.discard("")
    logos = {
        vendor: f"{data_path}/{LOGOS_DIR}/{VENDOR_LOGOS[vendor]}"
        for vendor in sorted(vendors)
        if vendor in VENDOR_LOGOS and _logo_resource(VENDOR_LOGOS[vendor]).is_file()
    }
    board_json = {
        "version": version,
        "generated": generated or default_generated(),
        "harness": {"name": HARNESS_NAME, "version": harness_version(), "href": HARNESS_HREF},
        "dataset": {
            "repo_id": source.get("repo_id"),
            "revision": source.get("revision"),
            "digest": first["dataset"]["digest"],
            "conversations": len(first["conversations"]),
            "tracks": first["tracks"]["total"],
            "reference_words": first["wer"]["reference_length"],
            "reference_characters": first["cer"]["reference_length"],
        },
        "profile": first["profile"],
        "conversations": _conversation_meta(first, board, titles),
        "transcriptsPath": data_path,
        "runs": runs,
        "operatingPoints": extra,
        "logos": logos,
    }
    return {"slug": slug, "version": version, "board": board_json, "transcripts": transcripts, "excluded": excluded}


def _logo_resource(file_name: str):
    return resources.files("ocular_evals").joinpath("templates", LOGOS_DIR, file_name)


def _clear(directory: Path, suffixes: tuple[str, ...]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.iterdir():
        if old.is_file() and old.suffix in suffixes:
            old.unlink()


def write_export(export: dict[str, Any], out: Path) -> Path:
    """Write ``board.json``, one transcripts file per board run and the vendor marks under ``out``."""
    root = Path(out) / export["slug"] / f"v{export['version']}"
    root.mkdir(parents=True, exist_ok=True)
    board = export["board"]
    (root / BOARD_FILE).write_text(json.dumps(board, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    transcripts_dir = root / TRANSCRIPTS_DIR
    _clear(transcripts_dir, (".json",))
    for transcript in export["transcripts"]:
        path = transcripts_dir / f"{transcript['run']}.json"
        path.write_text(json.dumps(transcript, ensure_ascii=False) + "\n", encoding="utf-8")

    logos_dir = root / LOGOS_DIR
    _clear(logos_dir, (".svg", ".png"))
    for vendor in board["logos"]:
        file_name = VENDOR_LOGOS[vendor]
        with resources.as_file(_logo_resource(file_name)) as source:
            shutil.copyfile(source, logos_dir / file_name)
    return root


def format_table(export: dict[str, Any]) -> str:
    board = export["board"]
    lines = [
        f"{len(board['runs'])} board runs, {len(board['operatingPoints'])} operating points, "
        f"{len(board['logos'])} vendor marks"
    ]
    for run in board["runs"]:
        lines.append(
            f"  {run['wer']:5.2f}  {run['cer']:5.2f}  {run['access']:11}  {run['route']:10}  "
            f"{run['displayName']} ({run['vendor']})  lat={run['medianLatencyS']}  fail={run['failures']}"
        )
    for run in board["operatingPoints"]:
        lines.append(
            f"  {run['wer']:5.2f}  {run['cer']:5.2f}  {run['access']:11}  {run['route']:10}  "
            f"{run['id']}: {run['displayName']} ({run['vendor']})  lat={run['medianLatencyS']}  "
            f"fail={run['failures']}  [operating point]"
        )
    return "\n".join(lines)


__all__ = [
    "BOARD_FILE",
    "DEFAULT_EXPORT",
    "DEFAULT_EXTRA",
    "HARNESS_HREF",
    "LOGOS_DIR",
    "TRANSCRIPTS_DIR",
    "build_export",
    "convert_run",
    "default_generated",
    "format_table",
    "harness_version",
    "is_prerelease",
    "write_export",
]
