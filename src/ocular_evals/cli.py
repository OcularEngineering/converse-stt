"""Command line entry point: ``ocular-evals``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ocular_evals import __version__
from ocular_evals.config import ConfigError, builtin_model_names, load_model_config
from ocular_evals.env import load_dotenv
from ocular_evals.metrics import percent, score_text
from ocular_evals.registry import (
    DEFAULT_PROVIDER,
    DEFAULT_REGISTRY,
    Registry,
    looks_like_env_name,
    suggest_env_name,
    upsert_dotenv,
)
from ocular_evals.text import DEFAULT_PROFILE, PROFILES, normalize

DEFAULT_DATASET = Path("data/converse-en-sample")
DEFAULT_RUNS = Path("runs")
DEFAULT_REPORT = Path("reports/converse-stt.html")
OPENROUTER_URL = "https://openrouter.ai/api/v1"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
MP3_UPLOAD_HOSTS = ("api.openai.com", "openrouter.ai")


def _fail(message: str) -> int:
    print(f"ocular-evals: error: {message}", file=sys.stderr)
    return 2


def _print_json(data) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


# --- dataset -----------------------------------------------------------------


def cmd_dataset_pull(args: argparse.Namespace) -> int:
    from ocular_evals.dataset import DatasetError, pull_dataset

    try:
        dataset = pull_dataset(args.output, repo_id=args.repo, revision=args.revision, token=args.token)
    except DatasetError as exc:
        return _fail(str(exc))
    print(f"{len(dataset.conversations)} conversations, {len(dataset.tracks)} tracks -> {dataset.root}")
    return 0


def cmd_dataset_build(args: argparse.Namespace) -> int:
    from ocular_evals.dataset import DatasetError, build_from_folder

    try:
        dataset = build_from_folder(args.source, args.output)
    except DatasetError as exc:
        return _fail(str(exc))
    print(f"{len(dataset.conversations)} conversations, {len(dataset.tracks)} tracks -> {dataset.root}")
    return 0


def cmd_dataset_verify(args: argparse.Namespace) -> int:
    from ocular_evals.dataset import DatasetError, load_dataset, verify_dataset

    try:
        dataset = load_dataset(args.dataset)
    except DatasetError as exc:
        return _fail(str(exc))
    problems = verify_dataset(dataset)
    for problem in problems:
        print(f"  - {problem}")
    if problems:
        return 1
    print(f"ok: {len(dataset.tracks)} tracks match the manifest (digest {dataset.digest()})")
    return 0


def cmd_dataset_show(args: argparse.Namespace) -> int:
    from ocular_evals.dataset import DatasetError, load_dataset

    try:
        dataset = load_dataset(args.dataset)
    except DatasetError as exc:
        return _fail(str(exc))
    source = dataset.source
    print(f"{dataset.root}  digest {dataset.digest()}")
    origin = source.get("repo_id") or source.get("path") or ""
    print(f"source: {source.get('kind')} {origin} {source.get('revision') or ''}".rstrip())
    total_words = 0
    for conversation in dataset.conversations:
        print(f"{conversation.id}  {conversation.language}  {conversation.duration_s / 60:.1f} min")
        for track in conversation.tracks:
            words = len(normalize(track.reference, language=conversation.language).split())
            total_words += words
            print(
                f"  {track.id}  speaker {track.speaker_id}  offset {track.offset_s:+.3f}s  "
                f"{track.duration_s:.1f}s  {words:,} words"
            )
    print(f"{len(dataset.tracks)} tracks, {total_words:,} normalized reference words")
    return 0


# --- models ------------------------------------------------------------------


def cmd_models_list(args: argparse.Namespace) -> int:
    registry = Registry.load(args.models_file)
    rows = []
    for name in registry.names():
        try:
            config = registry.config(name)
        except ConfigError as exc:
            rows.append((name, "invalid", str(exc), "", ""))
            continue
        base_url = config.options.get("base_url", "")
        rows.append((name, "models.toml", config.provider, config.model, base_url))
    for name in builtin_model_names():
        if name in registry:
            continue
        config = load_model_config(name, registry=args.models_file)
        rows.append((name, "built-in", config.provider, config.model, config.options.get("base_url", "")))
    if not rows:
        print("no models yet. Add one: ocular-evals models add <name> --url <base url> --key <ENV_NAME>")
        return 0
    width = max(len(row[0]) for row in rows)
    for name, source, provider, model, base_url in rows:
        tail = f"  {base_url}" if base_url else ""
        print(f"{name:{width}}  {source:11} {provider:18} {model}{tail}")
    if not registry.path.is_file():
        print(f"\n({registry.path} does not exist yet; `models add` creates it)")
    return 0


def _parse_options(items: list[str] | None) -> dict:
    options: dict = {}
    for item in items or ():
        if "=" not in item:
            raise ConfigError(f"--option expects KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        key, raw = key.strip(), raw.strip()
        lowered = raw.lower()
        if lowered in {"true", "false"}:
            options[key] = lowered == "true"
            continue
        try:
            options[key] = int(raw)
            continue
        except ValueError:
            pass
        try:
            options[key] = float(raw)
            continue
        except ValueError:
            pass
        options[key] = raw
    return options


def cmd_models_add(args: argparse.Namespace) -> int:
    from ocular_evals.providers import PROVIDERS, create_provider

    registry = Registry.load(args.models_file)
    provider_key = args.provider or DEFAULT_PROVIDER
    if provider_key not in PROVIDERS:
        return _fail(f"unknown provider {provider_key!r}; available: {', '.join(sorted(PROVIDERS))}")
    if args.openrouter:
        if provider_key != "openai-compatible":
            return _fail("--openrouter only applies to openai-compatible models")
        if args.url:
            return _fail("--openrouter and --url are mutually exclusive; OpenRouter's URL is implied")
        if not args.model or "/" not in args.model:
            return _fail("--openrouter needs the OpenRouter model id via --model, e.g. --model deepgram/nova-3")
        args.url = OPENROUTER_URL
        args.key = args.key or OPENROUTER_KEY_ENV
        args.notes = args.notes or "via OpenRouter"
        args.href = args.href or f"https://openrouter.ai/{args.model}"
    if provider_key in {"openai-compatible", "omi"} and not args.url:
        return _fail(
            "--url is required: the vendor's base URL, for example https://api.vendor.com/v1 "
            "(or pass --openrouter to route through OpenRouter)"
        )

    key_env: str | None = None
    secret: str | None = None
    if args.key == "-":
        args.key = sys.stdin.readline().strip()
    if args.key:
        if looks_like_env_name(args.key):
            key_env = args.key.strip()
        else:
            secret = args.key.strip()
            key_env = args.key_env or suggest_env_name(args.name)
    elif args.key_env:
        key_env = args.key_env
    elif provider_key != "faster-whisper":
        return _fail("--key is required: either the secret itself or the NAME of the variable that holds it in .env")

    options = _parse_options(args.option)
    if args.upload:
        options["upload"] = args.upload
    elif provider_key == "openai-compatible" and args.url and any(host in args.url for host in MP3_UPLOAD_HOSTS):
        options["upload"] = "mp3"  # these hosts cap multipart uploads at 25 MB; 15-minute FLAC tracks exceed it.
    if args.response_format:
        options["response_format"] = args.response_format

    try:
        config = registry.add(
            args.name,
            url=args.url,
            key_env=key_env,
            model=args.model,
            provider=provider_key,
            vendor=args.vendor,
            display_name=args.display_name,
            href=args.href,
            notes=args.notes,
            options=options or None,
            replace=args.replace,
        )
    except ConfigError as exc:
        return _fail(str(exc))

    if secret and key_env:
        upsert_dotenv(args.env_file, key_env, secret)
        os.environ[key_env] = secret
        print(f"stored the key in {args.env_file} as {key_env}")
    registry.save()
    print(f"added [{config.name}] to {registry.path}")

    problems = create_provider(config).check()
    if problems:
        print(f"{config.name}: not ready")
        for problem in problems:
            print(f"  - {problem}")
        if key_env and any(key_env in problem for problem in problems):
            print(f"  add a line to {args.env_file}:  {key_env}=<your key>")
        return 1
    print(f"{config.name}: ready ({config.provider}:{config.model}). Run it: ocular-evals run --model {config.name}")
    return 0


def cmd_models_show(args: argparse.Namespace) -> int:
    try:
        config = load_model_config(args.name, registry=args.models_file)
    except ConfigError as exc:
        return _fail(str(exc))
    data = config.to_dict()
    data["source"] = config.source
    _print_json(data)
    return 0


def cmd_models_remove(args: argparse.Namespace) -> int:
    registry = Registry.load(args.models_file)
    try:
        registry.remove(args.name)
    except ConfigError as exc:
        return _fail(str(exc))
    registry.save()
    print(f"removed [{args.name}] from {registry.path} (runs/{args.name} is untouched)")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    from ocular_evals.providers import create_provider

    names = list(args.model or []) + list(args.model_flag or [])
    if not names:
        names = Registry.load(args.models_file).names()
        if not names:
            return _fail("no models to check; name one or add some to models.toml")
    status = 0
    for value in names:
        try:
            config = load_model_config(value, registry=args.models_file)
            provider = create_provider(config)
        except ConfigError as exc:
            print(f"{value}: {exc}")
            status = 1
            continue
        problems = provider.check()
        if problems:
            status = 1
            print(f"{config.name}: not ready")
            for problem in problems:
                print(f"  - {problem}")
        else:
            print(f"{config.name}: ready ({config.provider}:{config.model})")
    return status


# --- run / score / report -------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    from ocular_evals.dataset import DatasetError, load_dataset
    from ocular_evals.runner import RunError, RunOptions, format_summary, score_run, transcribe_dataset

    try:
        config = load_model_config(args.model, registry=args.models_file)
        dataset = load_dataset(args.dataset)
    except (ConfigError, DatasetError) as exc:
        return _fail(str(exc))
    output = args.output or args.runs_dir / config.name
    try:
        options = RunOptions(
            profile=args.profile,
            resume=not args.fresh,
            conversation_ids=tuple(args.conversation or ()),
            track_ids=tuple(args.track or ()),
            max_consecutive_failures=args.max_failures,
        )
        counts = transcribe_dataset(dataset, config, output, options=options)
        print(
            f"{counts['transcribed']} transcribed, {counts['skipped']} already done, "
            f"{counts['failed']} failed of {counts['selected']} selected tracks",
            file=sys.stderr,
        )
        if counts["transcribed"] + counts["skipped"] == 0:
            return 1
        summary = score_run(output, dataset=dataset, profile=args.profile)
    except RunError as exc:
        return _fail(str(exc))
    print(format_summary(summary))
    print(f"run directory: {output}")
    return 0 if counts["failed"] == 0 else 1


def cmd_score(args: argparse.Namespace) -> int:
    from ocular_evals.dataset import DatasetError, load_dataset
    from ocular_evals.runner import RunError, format_summary, score_run

    try:
        dataset = load_dataset(args.dataset) if args.dataset else None
        summary = score_run(args.run, dataset=dataset, profile=args.profile)
    except (RunError, DatasetError) as exc:
        return _fail(str(exc))
    if args.json:
        _print_json(summary)
    else:
        print(format_summary(summary))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from ocular_evals.report import discover_runs, write_report
    from ocular_evals.runner import RunError

    run_dirs = list(args.run or [])
    if not run_dirs:
        run_dirs = discover_runs(args.runs_dir)
        if not run_dirs:
            return _fail(f"no scored runs under {args.runs_dir}; run a model first or pass --run")
    try:
        output = write_report(run_dirs, args.output, board=args.board, sample_label=args.label)
    except RunError as exc:
        return _fail(str(exc))
    print(f"{output}  ({len(run_dirs)} run{'s' if len(run_dirs) != 1 else ''})")
    return 0


# --- text ----------------------------------------------------------------------


def cmd_normalize(args: argparse.Namespace) -> int:
    text = " ".join(args.text) if args.text else sys.stdin.read()
    print(normalize(text, language=args.language, profile=args.profile))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    reference = args.reference.read_text(encoding="utf-8") if args.reference.exists() else args.reference.name
    hypothesis = args.hypothesis.read_text(encoding="utf-8") if args.hypothesis.exists() else args.hypothesis.name
    score = score_text(reference, hypothesis, language=args.language, profile=args.profile)
    print(f"WER {percent(score.wer.rate)}  {score.wer.to_dict()}")
    print(f"CER {percent(score.cer.rate)}  {score.cer.to_dict()}")
    if args.show:
        print(f"reference:  {score.reference}")
        print(f"hypothesis: {score.hypothesis}")
    return 0


# --- parser --------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ocular-evals",
        description="Ocular AI Converse-STT 2.0 harness: transcribe Ocular AI conversations and score WER / CER.",
    )
    parser.add_argument("--version", action="version", version=f"ocular-evals {__version__}")
    parser.add_argument("--env-file", type=Path, default=Path(".env"), help="dotenv file to load (default: ./.env)")
    parser.add_argument(
        "--models-file",
        type=Path,
        default=None,
        help=f"model registry (default: ./{DEFAULT_REGISTRY} or $OCULAR_MODELS_FILE)",
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=DEFAULT_RUNS, help="where run directories live (default: runs/)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    dataset = sub.add_parser("dataset", help="download, build, inspect or verify a dataset").add_subparsers(
        dest="dataset_command", required=True
    )
    pull = dataset.add_parser("pull", help="download an Ocular AI dataset from Hugging Face")
    from ocular_evals.dataset import DEFAULT_REPO_ID, DEFAULT_REVISION

    pull.add_argument("--repo", default=DEFAULT_REPO_ID)
    pull.add_argument("--revision", default=DEFAULT_REVISION)
    pull.add_argument("--token", default=None, help="Hugging Face token (defaults to HF_TOKEN / cached login)")
    pull.add_argument("--output", type=Path, default=DEFAULT_DATASET)
    pull.set_defaults(func=cmd_dataset_pull)
    build = dataset.add_parser("build", help="build a dataset from a local AudioFolder export")
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.set_defaults(func=cmd_dataset_build)
    verify = dataset.add_parser("verify", help="check audio files against the manifest")
    verify.add_argument("dataset", type=Path, nargs="?", default=DEFAULT_DATASET)
    verify.set_defaults(func=cmd_dataset_verify)
    show = dataset.add_parser("show", help="list conversations, tracks and reference word counts")
    show.add_argument("dataset", type=Path, nargs="?", default=DEFAULT_DATASET)
    show.set_defaults(func=cmd_dataset_show)

    models = sub.add_parser("models", help="list, add, show or remove models in models.toml")
    models.set_defaults(func=cmd_models_list)
    models_sub = models.add_subparsers(dest="models_command")
    models_sub.add_parser("list", help="list registered and built-in models").set_defaults(func=cmd_models_list)
    add = models_sub.add_parser(
        "add",
        help="register a model: a name, the API base URL and the key are enough",
        description=(
            "Register a model in models.toml. For an OpenAI-compatible endpoint only --url and --key are needed. "
            "--key accepts either the secret itself (stored in .env, never in models.toml) or the NAME of an "
            "environment variable that already holds it. Use `--key -` to read the secret from stdin and keep it "
            "out of your shell history."
        ),
    )
    add.add_argument("name", help="short id, e.g. gpt-4o-mini-transcribe (also the default model id and run dir)")
    add.add_argument("--url", help="API base URL, e.g. https://api.vendor.com/v1 (/audio/transcriptions is implied)")
    add.add_argument(
        "--openrouter",
        action="store_true",
        help=f"route through OpenRouter ({OPENROUTER_URL}) with --model <vendor/model>; key defaults to "
        f"{OPENROUTER_KEY_ENV}",
    )
    add.add_argument("--key", help="the API key, or the NAME of the .env variable holding it, or '-' for stdin")
    add.add_argument("--key-env", help="variable name to store a literal --key under (default: <VENDOR>_API_KEY)")
    add.add_argument("--model", help="model id sent to the API when it differs from the name")
    add.add_argument("--provider", choices=("openai-compatible", "omi", "faster-whisper"), default=None)
    add.add_argument("--vendor", help="shown on the leaderboard, e.g. OpenAI")
    add.add_argument("--display-name", help="shown on the leaderboard, e.g. 'GPT-4o mini Transcribe'")
    add.add_argument("--href", help="link for the leaderboard row")
    add.add_argument("--notes")
    add.add_argument(
        "--upload",
        choices=("flac", "wav16k", "mp3"),
        help="audio format to send (default flac; mp3 for api.openai.com)",
    )
    add.add_argument("--response-format", choices=("json", "verbose_json", "text"))
    add.add_argument("--option", action="append", metavar="KEY=VALUE", help="extra provider option (repeatable)")
    add.add_argument("--replace", action="store_true", help="overwrite an existing entry with the same name")
    add.set_defaults(func=cmd_models_add)
    show = models_sub.add_parser("show", help="print the resolved config for one model")
    show.add_argument("name")
    show.set_defaults(func=cmd_models_show)
    remove = models_sub.add_parser("remove", help="delete a model from models.toml")
    remove.add_argument("name")
    remove.set_defaults(func=cmd_models_remove)

    check = sub.add_parser("check", help="verify credentials and dependencies (default: every model in models.toml)")
    check.add_argument("model", nargs="*", help="model name or path to a TOML config")
    check.add_argument("--model", dest="model_flag", action="append", help="same as the positional form")
    check.set_defaults(func=cmd_check)

    run = sub.add_parser("run", help="transcribe the dataset with a model and score it")
    run.add_argument("--model", required=True, help="built-in model name or path to a TOML config")
    run.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    run.add_argument("--output", type=Path, default=None, help="run directory (default: runs/<model name>)")
    run.add_argument("--profile", choices=PROFILES, default=DEFAULT_PROFILE)
    run.add_argument("--conversation", action="append", help="only these conversation ids (repeatable)")
    run.add_argument("--track", action="append", help="only these track ids, e.g. ch1 (repeatable)")
    run.add_argument("--fresh", action="store_true", help="discard previous hypotheses in the output directory")
    run.add_argument("--max-failures", type=int, default=3, help="stop after this many consecutive failures")
    run.set_defaults(func=cmd_run)

    score = sub.add_parser("score", help="rescore an existing run (for example with another profile)")
    score.add_argument("--run", type=Path, required=True)
    score.add_argument("--dataset", type=Path, default=None, help="override the dataset recorded in run.json")
    score.add_argument("--profile", choices=PROFILES, default=None)
    score.add_argument("--json", action="store_true")
    score.set_defaults(func=cmd_score)

    report = sub.add_parser("report", help="build the leaderboard HTML from scored runs (default: every run in runs/)")
    report.add_argument(
        "--run", type=Path, action="append", help="run directory (repeatable); default: every scored run"
    )
    report.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    report.add_argument(
        "--board",
        default=None,
        help="also show a published board next to the runs: built-in name (converse-stt = the 1.0 figures) "
        "or a JSON path. Off by default.",
    )
    report.add_argument("--label", default="This sample", help="label for harness rows")
    report.set_defaults(func=cmd_report)

    norm = sub.add_parser("normalize", help="show how text is normalized before scoring")
    norm.add_argument("text", nargs="*")
    norm.add_argument("--language", default="en")
    norm.add_argument("--profile", choices=PROFILES, default=DEFAULT_PROFILE)
    norm.set_defaults(func=cmd_normalize)

    compare = sub.add_parser("compare", help="score one hypothesis against one reference (text or files)")
    compare.add_argument("reference", type=Path)
    compare.add_argument("hypothesis", type=Path)
    compare.add_argument("--language", default="en")
    compare.add_argument("--profile", choices=PROFILES, default=DEFAULT_PROFILE)
    compare.add_argument("--show", action="store_true", help="print the normalized texts")
    compare.set_defaults(func=cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    load_dotenv(args.env_file)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
