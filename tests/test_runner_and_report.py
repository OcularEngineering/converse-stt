from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ocular_evals import __version__
from ocular_evals.config import parse_model_config
from ocular_evals.dataset import build_dataset, load_dataset, read_jsonl
from ocular_evals.export import build_export, harness_version, write_export
from ocular_evals.providers.base import Hypothesis, Provider, ProviderError
from ocular_evals.registry import Registry
from ocular_evals.report import build_payload, load_board, render_html, write_report
from ocular_evals.runner import (
    FAILURES_FILE,
    HYPOTHESES_FILE,
    SCORES_FILE,
    SUMMARY_FILE,
    RunError,
    RunOptions,
    score_run,
    select_tracks,
    transcribe_dataset,
)

CONFIG_TEXT = """
name = "fake-model"
display_name = "Fake Model"
vendor = "Tests"
provider = "fake"
model = "fake-1"
href = "https://example.com"
"""
CONFIG = parse_model_config(CONFIG_TEXT)

REFERENCES = {
    ("s1", "a"): "Hello there, <breath/> how are you doing today?",
    ("s1", "b"): "I'm doing fine, thanks. <laugh/> And you?",
    ("s2", "a"): "Twenty-one people came to the meeting.",
    ("s2", "b"): "That is a lot <spn/> of people.",
}
ANSWERS = {
    ("s1", "ch1"): "hello there how are you doing today",
    ("s1", "ch2"): "I am doing fine thanks and you",
    ("s2", "ch1"): "21 people came to the meeting",
    ("s2", "ch2"): "that is lots of people",
}


class FakeProvider(Provider):
    key = "fake"

    def __init__(self, config, *, fail=(), answers=None):
        super().__init__(config)
        self.fail = set(fail)
        self.answers = answers or ANSWERS
        self.calls = []

    def check(self):
        return []

    def transcribe(self, audio: Path, *, language: str) -> Hypothesis:
        conversation = audio.parent.name
        track = audio.stem
        self.calls.append((conversation, track))
        if (conversation, track) in self.fail:
            raise ProviderError("boom sk-abcdefghijklmnopqrstuvwxyz", retryable=True, status=429)
        text = self.answers[(conversation, track)]
        return Hypothesis(text=text, language=language, segments=({"start_s": 0.0, "end_s": 1.0, "text": text},))


@pytest.fixture
def dataset(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    for (session, speaker), transcript in REFERENCES.items():
        offset = 0 if speaker == "a" else 250
        file_name = f"{1000 + offset}-{speaker}-hifi-audio-{1000 + offset}.flac"
        sf.write(source / file_name, np.zeros(8000, dtype="float32"), 16000, format="FLAC")
        rows.append(
            {
                "session_id": session,
                "speaker_id": speaker,
                "file_name": file_name,
                "language": "en",
                "language_locale": "en-US",
                "transcript": transcript,
                "segments": [{"index": 0, "start": 0.0, "end": 0.5, "text": transcript, "words": []}],
            }
        )
    return build_dataset(rows, lambda name: source / name, tmp_path / "dataset", source={"kind": "test"})


def test_select_tracks_filters(dataset):
    assert len(select_tracks(dataset, RunOptions())) == 4
    assert len(select_tracks(dataset, RunOptions(conversation_ids=("s1",)))) == 2
    assert len(select_tracks(dataset, RunOptions(track_ids=("ch2",)))) == 2
    with pytest.raises(RunError):
        select_tracks(dataset, RunOptions(conversation_ids=("nope",)))


def test_run_scores_and_resumes(dataset, tmp_path):
    output = tmp_path / "run"
    provider = FakeProvider(CONFIG)
    counts = transcribe_dataset(dataset, CONFIG, output, provider=provider, log=lambda _: None)
    assert counts == {"transcribed": 4, "skipped": 0, "failed": 0, "selected": 4}
    summary = score_run(output, dataset=dataset)

    assert summary["tracks"] == {"scored": 4, "missing": 0, "total": 4}
    # s1: perfect on both tracks. s2 ch2: "a lot" -> "lots" = 1 substitution + 1 deletion on 6 words.
    assert summary["wer"]["errors"] == 2
    assert summary["wer"]["reference_length"] == 7 + 7 + 6 + 6
    assert summary["cer"]["errors"] > 0
    by_id = {row["conversation"]: row for row in summary["conversations"]}
    assert by_id["s1"]["wer"]["errors"] == 0
    assert by_id["s2"]["wer"]["errors"] == 2
    assert by_id["s2"]["complete"] is True

    rows = read_jsonl(output / SCORES_FILE)
    assert sum(row["level"] == "track" for row in rows) == 4
    assert sum(row["level"] == "conversation" for row in rows) == 2
    tracks = {(row["conversation"], row["track"]): row for row in rows if row["level"] == "track"}
    track = tracks[("s2", "ch1")]
    assert track["reference"] == "21 people came to the meeting"
    assert track["hypothesis"] == "21 people came to the meeting"

    # Resume: nothing is transcribed again.
    again = FakeProvider(CONFIG)
    counts = transcribe_dataset(dataset, CONFIG, output, provider=again, log=lambda _: None)
    assert counts["skipped"] == 4 and again.calls == []
    assert (output / SUMMARY_FILE).exists()


def test_run_record_keeps_the_dataset_root_portable(dataset, tmp_path, monkeypatch):
    """A run made from the project directory records a relative root, so run.json carries no user paths."""
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "run"
    transcribe_dataset(dataset, CONFIG, output, provider=FakeProvider(CONFIG), log=lambda _: None)
    record = json.loads((output / "run.json").read_text())
    assert record["dataset"]["root"] == "dataset"
    assert score_run(output)["tracks"]["scored"] == 4, "re-scoring resolves the relative root from the same directory"

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    transcribe_dataset(dataset, CONFIG, tmp_path / "run2", provider=FakeProvider(CONFIG), log=lambda _: None)
    record = json.loads((tmp_path / "run2" / "run.json").read_text())
    assert Path(record["dataset"]["root"]).is_absolute(), "a dataset outside the working directory stays absolute"


def test_failures_are_recorded_redacted_and_stop_after_limit(dataset, tmp_path):
    output = tmp_path / "run"
    provider = FakeProvider(CONFIG, fail={("s1", "ch1"), ("s1", "ch2"), ("s2", "ch1")})
    options = RunOptions(max_consecutive_failures=2)
    counts = transcribe_dataset(dataset, CONFIG, output, options=options, provider=provider, log=lambda _: None)
    assert counts["failed"] == 2 and counts["transcribed"] == 0
    failures = read_jsonl(output / FAILURES_FILE)
    assert len(failures) == 2
    assert "sk-abc" not in json.dumps(failures)
    assert failures[0]["retryable"] is True and failures[0]["status"] == 429

    # A later run picks up the remaining tracks and scoring reports what is missing.
    provider = FakeProvider(CONFIG, fail={("s1", "ch1")})
    counts = transcribe_dataset(dataset, CONFIG, output, options=options, provider=provider, log=lambda _: None)
    assert counts["transcribed"] == 3 and counts["failed"] == 1
    summary = score_run(output, dataset=dataset)
    assert summary["tracks"]["missing"] == 1
    s1 = next(row for row in summary["conversations"] if row["conversation"] == "s1")
    assert s1["complete"] is False and s1["tracks"] == 1


def test_refuses_to_resume_a_different_model(dataset, tmp_path):
    output = tmp_path / "run"
    transcribe_dataset(dataset, CONFIG, output, provider=FakeProvider(CONFIG), log=lambda _: None)
    other = parse_model_config('name = "fake-model"\nprovider = "fake"\nmodel = "fake-2"\n')
    with pytest.raises(RunError, match="different model config"):
        transcribe_dataset(dataset, other, output, provider=FakeProvider(other), log=lambda _: None)
    counts = transcribe_dataset(
        dataset, other, output, options=RunOptions(resume=False), provider=FakeProvider(other), log=lambda _: None
    )
    assert counts["transcribed"] == 4
    assert all(row["model_digest"] == other.digest() for row in read_jsonl(output / HYPOTHESES_FILE))


def test_rescoring_with_plain_profile_changes_normalization(dataset, tmp_path):
    output = tmp_path / "run"
    transcribe_dataset(dataset, CONFIG, output, provider=FakeProvider(CONFIG), log=lambda _: None)
    whisper = score_run(output, dataset=dataset, profile="whisper")
    plain = score_run(output, dataset=dataset, profile="plain")
    assert whisper["wer"]["errors"] == 2
    # plain keeps "I'm" and "twenty-one" so the fake hypotheses no longer match.
    assert plain["wer"]["errors"] > whisper["wer"]["errors"]
    assert load_dataset(dataset.root).digest() == dataset.digest()


def test_report_payload_and_html(dataset, tmp_path):
    output = tmp_path / "run"
    transcribe_dataset(dataset, CONFIG, output, provider=FakeProvider(CONFIG), log=lambda _: None)
    score_run(output, dataset=dataset)
    board = load_board("converse-stt")
    assert board["models"][0]["rank"] == 1 and board["updated"] == "September 2026"
    assert load_board(None)["models"] == []

    payload = build_payload([output], board=board)
    assert payload["published"], "an explicit board shows published rows"
    run = payload["runs"][0]
    assert run["name"] == "Fake Model" and run["tracks"] == 4
    assert run["wer"] == pytest.approx(round(2 / 26 * 100, 2))
    assert run["cer"] is not None
    assert len(run["conversations"]) == 2
    channel = run["conversations"][0]["channels"][0]
    assert {"wer", "cer", "hits", "reference", "hypothesis", "repeated"} <= set(channel)
    assert run["conversations"][0]["title"].startswith("Hello there how are you")

    html = render_html(payload)
    assert "/*__DATA__*/ null" not in html and "Fake Model" in html and "</script>" in html
    assert "\\u003c" in html or "<script" in html

    page = write_report([output], tmp_path / "reports" / "board.html")
    html = page.read_text(encoding="utf-8")
    assert page.exists() and html.startswith("<!DOCTYPE html>")
    # Default report: only harness runs, no published rows.
    assert '"published": []' in html and "Reson8" not in html


def replay_ops(ops: list[list]) -> tuple[int, int]:
    """How many reference and hypothesis tokens a channel's ``ops`` runs consume."""
    ref_used = sum(count for op, count in ops if op in {"e", "s", "d"})
    hyp_used = sum(count for op, count in ops if op in {"e", "s", "i"})
    return ref_used, hyp_used


def test_report_payload_carries_review_fields(dataset, tmp_path):
    output = tmp_path / "run"
    transcribe_dataset(dataset, CONFIG, output, provider=FakeProvider(CONFIG), log=lambda _: None)
    summary = score_run(output, dataset=dataset)

    payload = build_payload([output], board=load_board("converse-stt"))
    run = payload["runs"][0]

    # Run level: provider, route and options come from the model config; the fake provider is neither local nor
    # routed through OpenRouter, is not in the board's openWeights/streaming lists and has no huggingface href.
    assert run["provider"] == "fake" and run["route"] == "direct" and run["options"] == {}
    assert run["access"] == "proprietary" and run["streaming"] is False
    assert run["scoredAt"] == summary["scored_at"] and isinstance(run["scoredAt"], str)
    assert (run["substitutions"], run["deletions"], run["insertions"]) == (1, 1, 0)
    assert run["substitutions"] + run["deletions"] + run["insertions"] == run["numerator"] == 2
    assert run["cerSubstitutions"] + run["cerDeletions"] + run["cerInsertions"] == run["cerNumerator"]
    assert all(isinstance(run[key], int) for key in ("cerSubstitutions", "cerDeletions", "cerInsertions"))
    assert run["latencyS"] is None, "the fake provider reports no latency, so there is no median"

    # Dataset block describes the sample the first board run was scored on.
    assert payload["dataset"] == {
        "conversations": 2,
        "tracks": 4,
        "referenceWords": run["denominator"],
        "referenceCharacters": run["cerDenominator"],
    }
    assert payload["dataset"]["referenceWords"] == 26

    # Conversation level: edit counts, duration spanning the offset channel, and title fallbacks for topic/speakers.
    by_id = {item["id"]: item for item in run["conversations"]}
    s2 = by_id["s2"]
    assert (s2["substitutions"], s2["deletions"], s2["insertions"]) == (1, 1, 0)
    assert s2["durationS"] == pytest.approx(0.75), "ch2 starts at 0.25 s and runs 0.5 s"
    assert s2["topic"] == s2["title"] and s2["speakers"] == {}
    assert by_id["s1"]["topic"] == by_id["s1"]["title"] and by_id["s1"]["speakers"] == {}

    # Channel level: full normalized texts, raw hypothesis as returned, and ops that replay to both lengths.
    channels = {(item["id"], c["channel"]): c for item in run["conversations"] for c in item["channels"]}
    worst = channels[("s2", "ch2")]
    assert worst["reference"] == "that is a lot of people"
    assert worst["hypothesis"] == worst["raw"] == "that is lots of people"
    assert worst["ops"] == [["e", 2], ["s", 1], ["d", 1], ["e", 2]]
    assert (worst["substitutions"], worst["deletions"], worst["insertions"]) == (1, 1, 0)
    polite = channels[("s1", "ch2")]
    assert polite["raw"] == "I am doing fine thanks and you", "raw keeps the provider's casing"
    assert polite["hypothesis"] == "i am doing fine thanks and you"
    for channel in channels.values():
        assert all(isinstance(op, list) and len(op) == 2 for op in channel["ops"])
        assert replay_ops(channel["ops"]) == (len(channel["reference"].split()), len(channel["hypothesis"].split()))
        assert sum(count for op, count in channel["ops"] if op == "e") == channel["hits"]


def test_board_metadata_marks_runs_and_conversations(dataset, tmp_path):
    output = tmp_path / "run"
    transcribe_dataset(dataset, CONFIG, output, provider=FakeProvider(CONFIG), log=lambda _: None)
    score_run(output, dataset=dataset)
    board = dict(load_board("converse-stt"))
    assert {"conversations", "openWeights", "streaming"} <= set(board)
    board["conversations"] = {
        **board["conversations"],
        "s1": {"topic": "Greetings", "speakers": {"ch1": "Male speaker", "ch2": "Female speaker"}},
    }
    board["openWeights"] = [*board["openWeights"], "fake-model"]
    board["streaming"] = [*board["streaming"], "fake-model"]

    payload = build_payload([output], board=board)
    run = payload["runs"][0]
    assert run["access"] == "open" and run["streaming"] is True
    assert run["route"] == "direct", "being listed as open weights does not change the route"
    by_id = {item["id"]: item for item in run["conversations"]}
    assert by_id["s1"]["topic"] == "Greetings"
    assert by_id["s1"]["speakers"] == {"ch1": "Male speaker", "ch2": "Female speaker"}
    assert by_id["s2"]["topic"] == by_id["s2"]["title"] and by_id["s2"]["speakers"] == {}
    assert payload["board"]["conversations"]["s1"]["topic"] == "Greetings"
    assert "models" not in payload["board"]


def test_openrouter_and_huggingface_runs_are_classified(dataset, tmp_path):
    routed = parse_model_config(
        CONFIG_TEXT.replace('name = "fake-model"', 'name = "fake-routed"').replace(
            'href = "https://example.com"',
            'href = "https://huggingface.co/tests/fake"\n[options]\nbase_url = "https://openrouter.ai/api/v1"\n'
            'api_key_env = "FAKE_API_KEY"\ntemperature = 0',
        )
    )
    output = tmp_path / "run"
    transcribe_dataset(dataset, routed, output, provider=FakeProvider(routed), log=lambda _: None)
    score_run(output, dataset=dataset)

    run = build_payload([output], board=load_board("converse-stt"))["runs"][0]
    assert run["route"] == "openrouter"
    assert run["options"] == {"base_url": "https://openrouter.ai/api/v1", "temperature": 0}, "*_env keys are dropped"
    assert run["access"] == "open", "weights hosted on huggingface count as open"
    assert run["streaming"] is False


def test_vendor_marks_are_embedded_as_data_urls(dataset, tmp_path):
    from ocular_evals.report import vendor_logo

    assert vendor_logo("Tests") is None, "vendors without a shipped mark fall back to their initial"
    nvidia = vendor_logo("NVIDIA")
    assert nvidia is not None and nvidia.startswith("data:image/svg+xml;base64,")
    assert vendor_logo("Cohere").startswith("data:image/png;base64,")

    output = tmp_path / "run"
    transcribe_dataset(dataset, CONFIG, output, provider=FakeProvider(CONFIG), log=lambda _: None)
    score_run(output, dataset=dataset)
    payload = build_payload([output], board=load_board("converse-stt"))
    # Only vendors that appear on the page are embedded; published Version 1 vendors count.
    assert "Tests" not in payload["logos"] and payload["logos"]["Reson8"].startswith("data:image/png;base64,")
    assert "Hexgrad" not in payload["logos"]


def test_grouped_runs_land_on_the_version_1_tab(dataset, tmp_path):
    board_run = tmp_path / "board"
    transcribe_dataset(dataset, CONFIG, board_run, provider=FakeProvider(CONFIG), log=lambda _: None)
    score_run(board_run, dataset=dataset)
    v1_config = parse_model_config(
        CONFIG_TEXT.replace('name = "fake-model"', 'name = "gpt-4o-transcribe"').replace(
            'vendor = "Tests"', 'vendor = "Tests"\ngroup = "converse-stt-1.0"'
        )
    )
    v1_run = tmp_path / "v1"
    transcribe_dataset(dataset, v1_config, v1_run, provider=FakeProvider(v1_config), log=lambda _: None)
    score_run(v1_run, dataset=dataset)

    payload = build_payload([board_run, v1_run], board=load_board("converse-stt"))

    assert [run["id"] for run in payload["runs"]] == ["fake-model"], "grouped runs stay off the 2.0 board"
    assert [run["id"] for run in payload["groupRuns"]["converse-stt-1.0"]] == ["gpt-4o-transcribe"]
    matched = {model["name"]: model.get("rerun") for model in payload["published"]}
    assert matched["GPT-4o Transcribe"] == "gpt-4o-transcribe"
    assert matched["Gradium"] is None and "blocked" in next(m for m in payload["published"] if m["name"] == "Gradium")
    assert payload["groupExtra"] == []
    html = render_html(payload)
    assert "Version 1 models" in html and '"group": "converse-stt-1.0"' in html


def test_damaged_audio_is_never_sent_to_the_provider(dataset, tmp_path):
    damaged = dataset.root / "audio" / "s2" / "ch1.flac"
    damaged.write_bytes(b"not the audio in the manifest")
    provider = FakeProvider(CONFIG)
    output = tmp_path / "run"

    counts = transcribe_dataset(dataset, CONFIG, output, provider=provider, log=lambda _: None)

    assert counts == {"transcribed": 3, "skipped": 0, "failed": 1, "selected": 4}
    assert ("s2", "ch1") not in provider.calls
    failure = read_jsonl(output / "failures.jsonl")[0]
    assert failure["track"] == "ch1" and "does not match the manifest" in failure["error"]
    assert failure["retryable"] is False


def test_concurrent_runs_on_one_directory_are_refused(dataset, tmp_path):
    import subprocess
    import sys

    from ocular_evals.runner import LOCK_FILE

    output = tmp_path / "run"
    output.mkdir()
    # Hold the lock from another process, the way a still-running `ocular-evals run` would.
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import fcntl,os,sys,time; fd=os.open(sys.argv[1], os.O_RDWR|os.O_CREAT); "
            "fcntl.flock(fd, fcntl.LOCK_EX); print('locked', flush=True); time.sleep(30)",
            str(output / LOCK_FILE),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "locked"
        with pytest.raises(RunError, match="already writing"):
            transcribe_dataset(dataset, CONFIG, output, provider=FakeProvider(CONFIG), log=lambda _: None)
        assert not (output / "hypotheses.jsonl").exists()
    finally:
        holder.kill()
        holder.wait()
    # Once the other process is gone the lock is free again.
    counts = transcribe_dataset(dataset, CONFIG, output, provider=FakeProvider(CONFIG), log=lambda _: None)
    assert counts["transcribed"] == 4


def _scored_run(dataset, output: Path, config=CONFIG) -> Path:
    transcribe_dataset(dataset, config, output, provider=FakeProvider(config), log=lambda _: None)
    score_run(output, dataset=dataset)
    return output


def _export(tmp_path, runs, extra=(), *, models_toml: str = "", **kwargs):
    models_file = tmp_path / "models.toml"
    models_file.write_text(models_toml, encoding="utf-8")
    return build_export(
        list(runs),
        list(extra),
        board=load_board("converse-stt"),
        registry=Registry.load(models_file),
        generated="Oct 21, 2026",
        **kwargs,
    )


def test_export_writes_board_transcripts_and_logos(dataset, tmp_path):
    nvidia_text = CONFIG_TEXT.replace("fake-model", "fake-nvidia").replace('vendor = "Tests"', 'vendor = "NVIDIA"')
    nvidia = parse_model_config(nvidia_text)
    runs = [
        _scored_run(dataset, tmp_path / "runs" / "fake-model"),
        _scored_run(dataset, tmp_path / "runs" / "nv", nvidia),
    ]

    export = _export(tmp_path, runs)
    board = export["board"]
    assert export["slug"] == "converse-stt" and export["version"] == "2", "slug and version come from the board file"
    assert list(board) == [
        "version",
        "generated",
        "harness",
        "dataset",
        "profile",
        "conversations",
        "transcriptsPath",
        "runs",
        "operatingPoints",
        "logos",
    ]
    assert board["version"] == "2" and board["generated"] == "Oct 21, 2026"
    assert board["harness"] == {
        "name": "ocular-evals",
        "version": harness_version(),
        "href": "https://github.com/OcularEngineering/converse-stt",
    }
    assert board["dataset"] == {
        "repo_id": None,
        "revision": None,
        "digest": dataset.digest(),
        "conversations": 2,
        "tracks": 4,
        "reference_words": 26,
        "reference_characters": board["runs"][0]["counts"]["cer"]["reference_length"],
    }
    assert board["profile"] == "whisper" and board["transcriptsPath"] == "/data/converse-stt/v2"
    assert board["operatingPoints"] == []
    # Marks for the harness runs' vendors plus the published Version 1 vendors the site still shows;
    # "Tests" has no mark and is left out.
    assert board["logos"]["NVIDIA"] == "/data/converse-stt/v2/logos/nvidia.svg"
    assert "Tests" not in board["logos"]
    published_vendors = {model["vendor"] for model in load_board("converse-stt")["models"]}
    assert published_vendors & {"Cartesia", "Reson8"} <= set(board["logos"]), "Version 1 vendors get marks too"

    meta = board["conversations"][0]
    assert set(meta) == {"id", "short", "topic", "speakers", "duration_s", "reference_words", "tracks"}
    assert meta["id"] == meta["short"] == "s1" and meta["speakers"] == {}
    assert meta["topic"].startswith("Hello there how are you"), "no board description: the first words stand in"
    assert [track["track"] for track in meta["tracks"]] == ["ch1", "ch2"] and meta["reference_words"] == 14

    assert [run["id"] for run in board["runs"]] == ["fake-model", "fake-nvidia"], "equal WER: ordered by id"
    run = board["runs"][0]
    assert set(run) == {
        "id",
        "displayName",
        "vendor",
        "model",
        "href",
        "provider",
        "route",
        "access",
        "streaming",
        "notes",
        "options",
        "wer",
        "cer",
        "counts",
        "medianLatencyS",
        "tracksScored",
        "tracksTotal",
        "failures",
        "complete",
        "scoredAt",
        "modelDigest",
        "harness",
        "conversations",
    }
    assert run["displayName"] == "Fake Model" and run["wer"] == pytest.approx(round(2 / 26 * 100, 2))
    assert run["route"] == "direct" and run["access"] == "proprietary" and run["streaming"] is False
    assert run["counts"]["wer"]["errors"] == 2 and run["counts"]["wer"]["rate"] == run["wer"]
    assert run["medianLatencyS"] < 0.1, "the fake provider answers instantly; the recorded latency is kept, not dropped"
    assert run["complete"] is True and run["tracksScored"] == 4
    assert run["harness"] == {"name": "ocular-evals", "version": __version__}
    conversation = run["conversations"][1]
    assert set(conversation) == {"id", "complete", "duration_s", "wer", "cer", "counts", "tracks"}
    assert conversation["duration_s"] == pytest.approx(0.75)
    assert set(conversation["tracks"][0]) == {"track", "duration_s", "offset_s", "latency_s", "wer", "cer"}

    root = write_export(export, tmp_path / "export")
    assert root == tmp_path / "export" / "converse-stt" / "v2"
    written = json.loads((root / "board.json").read_text(encoding="utf-8"))
    assert written == board
    assert sorted(path.name for path in (root / "transcripts").iterdir()) == ["fake-model.json", "fake-nvidia.json"]
    assert (root / "logos" / "nvidia.svg").read_bytes().startswith(b"<")
    transcript = json.loads((root / "transcripts" / "fake-model.json").read_text(encoding="utf-8"))
    assert list(transcript) == ["run", "model", "profile", "conversations"]
    assert transcript["run"] == "fake-model" and transcript["model"] == "Fake Model"
    assert transcript["profile"] == "whisper"
    tracks = {(c["id"], t["track"]): t for c in transcript["conversations"] for t in c["tracks"]}
    assert set(tracks[("s2", "ch2")]) == {"track", "reference", "hypothesis", "raw", "ops"}
    assert tracks[("s2", "ch2")]["ops"] == [["e", 2], ["s", 1], ["d", 1], ["e", 2]]
    assert tracks[("s1", "ch2")]["raw"] == "I am doing fine thanks and you"
    for track in tracks.values():
        assert replay_ops(track["ops"]) == (len(track["reference"].split()), len(track["hypothesis"].split()))

    # Re-exporting removes transcripts of runs that are gone.
    (root / "transcripts" / "stale.json").write_text("{}", encoding="utf-8")
    write_export(export, tmp_path / "export")
    assert not (root / "transcripts" / "stale.json").exists()


def test_export_excludes_prerelease_models_unless_asked(dataset, tmp_path):
    early = parse_model_config(CONFIG_TEXT.replace('name = "fake-model"', 'name = "fake-early"\nprerelease = true'))
    stable = _scored_run(dataset, tmp_path / "runs" / "fake-model")
    flagged = _scored_run(dataset, tmp_path / "runs" / "fake-early", early)
    assert json.loads((flagged / SUMMARY_FILE).read_text())["model"]["prerelease"] is True

    export = _export(tmp_path, [stable, flagged])
    assert [run["id"] for run in export["board"]["runs"]] == ["fake-model"]
    assert export["excluded"] == ["fake-early"] and [t["run"] for t in export["transcripts"]] == ["fake-model"]

    export = _export(tmp_path, [stable, flagged], include_prerelease=True)
    assert sorted(run["id"] for run in export["board"]["runs"]) == ["fake-early", "fake-model"]
    assert export["excluded"] == []

    # Runs recorded before the flag existed are still excluded when the current models.toml marks the model.
    other = _scored_run(
        dataset, tmp_path / "runs" / "other", parse_model_config(CONFIG_TEXT.replace("fake-model", "fake-other"))
    )
    marked = '[fake-model]\nprovider = "fake"\nprerelease = true\n'
    export = _export(tmp_path, [stable, flagged, other], models_toml=marked)
    assert [run["id"] for run in export["board"]["runs"]] == ["fake-other"]
    assert export["excluded"] == ["fake-model", "fake-early"]

    with pytest.raises(RunError, match=r"no runs to export \(1 pre-release run\(s\) excluded\)"):
        _export(tmp_path, [flagged])


def test_export_operating_points_come_from_the_extra_directory(dataset, tmp_path):
    board_run = _scored_run(dataset, tmp_path / "runs" / "fake-model")
    # The same model at another setting: a different directory name, the same model name.
    extra_run = _scored_run(dataset, tmp_path / "runs-extra" / "fake-model-fast")

    export = _export(tmp_path, [board_run], [extra_run])
    assert [run["id"] for run in export["board"]["runs"]] == ["fake-model"]
    points = export["board"]["operatingPoints"]
    assert [point["id"] for point in points] == ["fake-model-fast"], "operating points are keyed by directory"
    assert points[0]["displayName"] == "Fake Model" and set(points[0]) == set(export["board"]["runs"][0])
    assert [t["run"] for t in export["transcripts"]] == ["fake-model"], "operating points get no transcripts file"
    root = write_export(export, tmp_path / "export")
    assert not (root / "transcripts" / "fake-model-fast.json").exists()


def test_export_cli_end_to_end(dataset, tmp_path, monkeypatch, capsys):
    from ocular_evals.cli import main

    monkeypatch.chdir(tmp_path)
    _scored_run(dataset, tmp_path / "runs" / "fake-model")
    (tmp_path / "models.toml").write_text("", encoding="utf-8")
    assert main(["export", "--generated", "Oct 21, 2026"]) == 0
    out = capsys.readouterr().out
    assert "1 board runs, 0 operating points" in out and "Fake Model (Tests)" in out
    board = json.loads((tmp_path / "export" / "converse-stt" / "v2" / "board.json").read_text(encoding="utf-8"))
    assert board["generated"] == "Oct 21, 2026" and [run["id"] for run in board["runs"]] == ["fake-model"]
    assert (tmp_path / "export" / "converse-stt" / "v2" / "transcripts" / "fake-model.json").is_file()
    assert main(["export", "--runs", "nowhere"]) == 2


def test_discover_runs_finds_only_scored_directories(dataset, tmp_path):
    from ocular_evals.report import discover_runs

    runs = tmp_path / "runs"
    scored = runs / "fake"
    transcribe_dataset(dataset, CONFIG, scored, provider=FakeProvider(CONFIG), log=lambda _: None)
    score_run(scored, dataset=dataset)
    unscored = runs / "half"
    transcribe_dataset(dataset, CONFIG, unscored, provider=FakeProvider(CONFIG), log=lambda _: None)
    (runs / "notes.txt").write_text("x", encoding="utf-8")
    assert discover_runs(runs) == [scored]
    assert discover_runs(tmp_path / "missing") == []
