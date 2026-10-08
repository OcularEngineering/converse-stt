# Ocular AI Evals Research

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Ocular AI's open-source evaluation harness for the [Converse-STT](https://www.useocular.com/converse/stt)
board, released under the [MIT License](LICENSE).
It transcribes each speaker track of an Ocular AI full-duplex conversation with a speech-to-text model and
reports **word error rate (WER)** and **character error rate (CER)**.

The design borrows from the public research harnesses we compare against (separate channel per speaker,
summed edit counts, Whisper-style text normalization) but the implementation is independent and much
smaller: one package, one manifest format, one command.

## Quick start

```bash
cd "Ocular Evals Research"
uv sync --group dev              # hosted providers only
uv sync --group dev --extra local  # also installs faster-whisper for local Whisper runs
cp .env.example .env             # add HF_TOKEN and the provider keys you need, e.g. OPENAI_API_KEY

uv run ocular-evals dataset pull                 # Ocular AI English sample from Hugging Face -> data/converse-en-sample
uv run ocular-evals dataset show
uv run ocular-evals check                        # credentials + dependencies for every model in models.toml
uv run ocular-evals run --model gpt-4o-mini-transcribe   # transcribe + score -> runs/gpt-4o-mini-transcribe
uv run ocular-evals report                       # every scored run in runs/ -> reports/converse-stt.html
open reports/converse-stt.html
```

`.env` is loaded automatically from the working directory and never committed.

## Adding a model

Models live in `models.toml` at the project root. For any API that speaks the OpenAI
`POST /audio/transcriptions` protocol (OpenAI itself, the many hosted services that mirror it, or a
self-hosted server) a model is a URL and a key:

```bash
uv run ocular-evals models add <model-name> --url https://api.<provider>.com/v1 --key <your-api-key>
#   stored the key in .env as <PROVIDER>_API_KEY
#   added [<model-name>] to models.toml
#   <model-name>: ready (openai-compatible:<model-name>). Run it: ocular-evals run --model <model-name>
uv run ocular-evals run --model <model-name>
uv run ocular-evals report
```

For example, OpenAI's small transcription model:

```bash
uv run ocular-evals models add gpt-4o-mini-transcribe --url https://api.openai.com/v1 --key sk-... --vendor OpenAI
```

`--key` takes either the secret itself (written to `.env`, never to `models.toml`) or the NAME of a variable
that already holds it (`--key OPENAI_API_KEY`). Use `--key -` to paste the secret on stdin and keep it out of
shell history. Useful extras: `--model <api-model-id>` when the id the API expects differs from the name,
`--vendor` and `--display-name` for the leaderboard, `--upload mp3` when the API caps uploads (OpenAI's 25 MB
limit is handled automatically), `--provider faster-whisper` for local Whisper or another built-in provider id
for job-style APIs, `--option key=value` for anything else.

Editing the file by hand is equally fine:

```toml
[<model-name>]
url = "https://api.<provider>.com/v1"    # base URL; /audio/transcriptions is implied
key = "<PROVIDER>_API_KEY"               # NAME of the variable in .env
model = "<api-model-id>"                 # optional, defaults to the table name
vendor = "<Provider>"                    # optional, leaderboard label
```

`ocular-evals models` lists everything, `models show <name>` prints the resolved config, `models remove <name>`
drops an entry (its `runs/<name>` directory is kept). Names in `models.toml` shadow the built-in configs in
`src/ocular_evals/models/`.

## How a score is produced

1. **Dataset.** `dataset pull` downloads the Hugging Face AudioFolder at a pinned revision and writes
   `manifest.json`: one conversation per session, one track per speaker microphone, with the original
   transcript, word timings shifted onto the shared conversation timeline, the recording offset between the
   two microphones, and a SHA-256 of every FLAC. `dataset verify` re-checks the audio against the manifest.
2. **Transcription.** `run` sends each track to the provider named in the model config and appends the raw
   text to `hypotheses.jsonl`. Runs are resumable: tracks that already have a hypothesis for the same model
   config and audio hash are skipped. Failures are written to `failures.jsonl` with the API key redacted.
   A run directory is locked while a run is in progress, so starting the same model twice refuses instead of
   interleaving two transcriptions. Transient errors (429/5xx/timeouts) are retried with backoff; a 429 that
   reports exhausted credits or quota is reported once and not retried.
3. **Normalization.** Reference and hypothesis go through the same `whisper` profile: Ocular AI markup such as
   `<breath/>`, `<laugh>…</laugh>`, `<spn/>` and `<unclear/>` is removed, then the Whisper English text
   normalizer (contractions, numbers, spelling variants, casing), then punctuation becomes spaces. A `plain`
   profile (lowercase, strip punctuation, keep apostrophes and hyphens) is available for inspection.
4. **Scoring.** Each track is aligned with jiwer. WER counts whitespace tokens; CER counts non-space
   characters. Edit counts (hits, substitutions, deletions, insertions) are summed across both tracks of a
   conversation and across the corpus, so the headline number is `total errors / total reference words`,
   never an average of percentages. `scores.jsonl` keeps the normalized texts next to every count;
   `summary.json` holds the totals.
5. **Report.** `report` renders a self-contained HTML leaderboard from every scored run in `runs/`
   (or the `--run` directories you name), toggles between WER and CER, and shows reference and hypothesis
   side by side for every track. A repeated short segment is flagged because it usually means the decoder
   looped. Pass `--board converse-stt` to also show the published Converse-STT figures next to the runs.

## Model configs

`models.toml` is the short form described above. Under the hood every entry becomes a full model config;
built-in examples live in `src/ocular_evals/models/*.toml`, and any TOML file with that shape works with
`--model path/to/file.toml`:

```toml
name = "gpt-4o-mini-transcribe"
display_name = "GPT-4o mini Transcribe"
vendor = "OpenAI"
provider = "openai-compatible"   # openai-compatible | omi | faster-whisper
model = "gpt-4o-mini-transcribe"
env = ["OPENAI_API_KEY"]
href = "https://platform.openai.com/docs/guides/speech-to-text"

[options]
response_format = "json"
upload = "mp3"                   # OpenAI caps uploads at 25 MB
```

| provider            | what it calls                                                            | key options                                                                 |
| ------------------- | ------------------------------------------------------------------------ | --------------------------------------------------------------------------- |
| `openai-compatible` | `POST {base_url}/audio/transcriptions` (OpenAI and any compatible server) | `base_url`, `api_key_env`, `response_format`, `upload` (`flac`/`wav16k`/`mp3`) |
| `omi`               | asynchronous job API for long audio: create job, upload, poll for result | `base_url`, `api_key_env`, `response_format`, `poll_wait_s`, `deadline_s`   |
| `faster-whisper`    | local CTranslate2 Whisper                                                 | `device`, `compute_type`, `beam_size`, `vad_filter`, `condition_on_previous_text` |

Options never contain credentials; keys are read from the environment variables listed in `env`.

## Layout

```
models.toml       the models this harness runs (url + key per model)
src/ocular_evals/
  dataset.py      Hugging Face / local AudioFolder -> manifest.json
  text.py         normalization profiles
  metrics.py      EditCounts, WER / CER alignment
  config.py       model TOML loading
  registry.py     models.toml parsing / `models add`
  providers/      openai_compatible, omi, faster_whisper (+ retrying HTTP, audio prep)
  runner.py       resumable transcription + scoring of a run directory
  report.py       leaderboard HTML
  models/         built-in model configs
  boards/         published Converse-STT board used by the report
  templates/      leaderboard.html
tests/            pytest, no network
data/             datasets (gitignored)
runs/             run directories (gitignored)
reports/          generated HTML (gitignored)
```

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

## Roadmap

- **Diarization metrics.** Score who-spoke-when on the mixed two-speaker audio (DER, speaker-attributed
  WER) alongside the per-track WER / CER reported today.
- **More models.** Keep adding hosted and local speech-to-text models to `models.toml` so the board covers
  every provider we are asked about; contributions of new entries are welcome.
- **More providers.** Native adapters for APIs that do not speak the OpenAI transcription protocol
  (streaming / WebSocket vendors, batch job APIs) beyond the current `openai-compatible`, `omi` and
  `faster-whisper` backends.
- **More datasets and languages.** Run the same harness over additional Ocular AI conversation sets as they
  are published, including non-English ones.
- **Latency and cost.** Record time-to-transcript and per-minute pricing next to accuracy so the board can
  rank on more than error rate.
- **Published board sync.** Push verified harness results straight to the public
  [Converse-STT](https://www.useocular.com/converse/stt) leaderboard.

## License

Copyright (c) 2026 Ocular AI. This project is released under the [MIT License](LICENSE); you are free to
use, modify and redistribute it, including commercially, as long as the copyright and permission notice are
kept. Contributions are accepted under the same license.

The harness code is MIT; the Ocular AI conversation datasets it downloads from Hugging Face carry their own
terms, stated on each dataset card. Model outputs and scores you generate belong to you.
