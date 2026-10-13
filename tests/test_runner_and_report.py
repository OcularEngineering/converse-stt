from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ocular_evals.config import parse_model_config
from ocular_evals.dataset import build_dataset, load_dataset, read_jsonl
from ocular_evals.providers.base import Hypothesis, Provider, ProviderError
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
