# Converse-STT 2.0

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Ocular AI's open-source evaluation harness for **Converse-STT**, the speech-to-text benchmark in the
[Ocular AI Converse](https://www.useocular.com/converse/stt) family, released under the
[MIT License](LICENSE). It transcribes each speaker track of an Ocular AI full-duplex conversation with a
speech-to-text model and reports **word error rate (WER)** and **character error rate (CER)**.

This repository is Converse-STT **2.0**. The first edition was scored in collaboration with Cekura; 2.0 is
run independently by Ocular AI, and every number on the board is produced by the code in this repository,
so anyone can reproduce a score or add a model.

## About the benchmark

**Converse** is Ocular AI's family of benchmarks for AI on real-world, two-person conversation. **Converse-STT**,
the Conversational Speech-to-Text Benchmark, measures how accurately frontier speech-to-text models transcribe
real, unscripted American English conversations between two people. A lower word error rate is a more
accurate transcript.

**The recordings.** Ocular AI's Hi-Fi conversational dataset: four American English conversations of about
15 minutes each, roughly 60 minutes of speech across eight speaker tracks. Each speaker is recorded on a
separate microphone at 48 kHz and delivered as 24-bit mono FLAC with verbatim, time-aligned transcripts. The
recordings keep the pauses, fillers, repetitions and false starts of ordinary talk. A public sample with
recordings and reference transcripts is on Hugging Face; it is a sample of a larger collection that Ocular AI
licenses.

**The score.** Word error rate: incorrect, missing and extra words divided by the number of words in the
reference. Breath and laughter tags are removed before scoring, so a model is not penalized for leaving them
out. This harness reports character error rate alongside it and keeps every normalized reference and
hypothesis so a score can be read, not just quoted.

**Converse-STT 1.0 (with Cekura).** The first edition scored 15 speech-to-text models on the Hi-Fi
conversations and on public Pipecat audio. Twelve of the 15 had a higher error rate on the conversational
recordings, four more than doubled, and the leading model changed with the dataset: the model that led Pipecat
at 1.93% ranked ninth on the conversations at 4.23%, while the conversational leader scored 2.93%. Those
figures are kept in `src/ocular_evals/boards/converse-stt.json`. With `report --board converse-stt` the page
gets a **Version 1 models** tab that lists those 15 models with their 1.0 figure and, where the 1.0 model has
been re-run with this harness (matched through the board file's `run` field), the 2.0 score beside it; models
that could not be re-run say why. The re-runs are ranked on the 2.0 board as well; an entry tagged
`group = "converse-stt-1.0"` in `models.toml` would instead appear on the Version 1 tab only. Full write-up: [useocular.com/converse/stt](https://www.useocular.com/converse/stt).

**Converse-STT 2.0 (this repository).** Independent of any evaluation partner. Ocular AI runs every model
through this harness on the same Hi-Fi conversations, publishes the harness, the model configurations and the
dataset manifest, and adds models as they are released.

The design follows established conventions for conversational ASR evaluation (separate channel per speaker,
summed edit counts, Whisper-style text normalization), and the implementation is deliberately small: one
package, one manifest format, one command. Results and methodology are written up in
[`docs/converse-stt-2.0-technical-report.md`](docs/converse-stt-2.0-technical-report.md).

## Quick start

```bash
git clone https://github.com/OcularEngineering/converse-stt && cd converse-stt
uv sync --group dev              # hosted providers only
uv sync --group dev --extra local  # also installs faster-whisper for local Whisper runs
uv sync --group dev --extra nemotron  # torch + transformers for the local Transformers models (NVIDIA Nemotron, Cohere Transcribe)
cp .env.example .env             # add OPENROUTER_API_KEY and any direct vendor keys you use (HF_TOKEN is optional)

uv run ocular-evals dataset pull                 # Ocular AI English sample from Hugging Face -> data/converse-en-sample
uv run ocular-evals dataset show
uv run ocular-evals check                        # credentials + dependencies for every model in models.toml
uv run ocular-evals run --model gpt-4o-mini-transcribe   # transcribe + score -> runs/gpt-4o-mini-transcribe
uv run ocular-evals report                       # every scored run in runs/ -> reports/converse-stt.html
open reports/converse-stt.html
uv run ocular-evals export                       # board + transcripts as JSON for the website -> export/converse-stt/v2/
```

`.env` is loaded automatically from the working directory and never committed.

## Adding a model

Models live in `models.toml` at the project root. There are two ways to reach a hosted model; both end up as
the same kind of entry, and you can mix them freely.

### Through OpenRouter (one key, many models)

[OpenRouter](https://openrouter.ai/docs/guides/overview/multimodal/stt) exposes a growing catalog of
speech-to-text models behind one OpenAI-style `/audio/transcriptions` endpoint and one key. The entries in
`models.toml` marked `via OpenRouter` need nothing more than `OPENROUTER_API_KEY` in `.env`. To add another
model from the catalog, pass its OpenRouter id:

```bash
uv run ocular-evals models add <model-name> --openrouter --model <vendor>/<model-id> --vendor <Vendor>
uv run ocular-evals run --model <model-name>
```

OpenRouter's `/models` catalogue does not enumerate transcription models; the `/audio/transcriptions` endpoint answers
`Model <id> does not exist` for ids it does not serve, so a quick `ocular-evals run --model <name> --conversation <one id>`
is the reliable way to find out. OpenRouter may route a model id to more than one upstream host and includes the proxy hop in latency, so
every routed entry carries `notes = "via OpenRouter"` and the leaderboard can say so.

### Directly, with the vendor's URL and key

For any API that speaks the OpenAI `POST /audio/transcriptions` protocol (OpenAI itself, the many hosted
services that mirror it, or a self-hosted server) a model is a URL and a key:

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
`--vendor` and `--display-name` for the leaderboard, `--upload mp3` when the API caps uploads (the 25 MB
limits of OpenAI and OpenRouter are handled automatically), `--provider faster-whisper` for local Whisper, `--provider nemotron-asr` for NVIDIA's Nemotron
streaming models, `--provider cohere-transcribe` for Cohere Transcribe, or another built-in provider id for job-style APIs, `--option key=value` for anything else. To run the same model
both ways, give the direct entry its own name (or use `--replace` to switch an existing one).

Editing the file by hand is equally fine:

```toml
[<model-name>]                           # direct
url = "https://api.<provider>.com/v1"    # base URL; /audio/transcriptions is implied
key = "<PROVIDER>_API_KEY"               # NAME of the variable in .env
model = "<api-model-id>"                 # optional, defaults to the table name
vendor = "<Provider>"                    # optional, leaderboard label

[<model-name>-openrouter]                # the same model through OpenRouter
url = "https://openrouter.ai/api/v1"
key = "OPENROUTER_API_KEY"
model = "<vendor>/<model-id>"
vendor = "<Provider>"
notes = "via OpenRouter"
upload = "mp3"
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
   looped. Pass `--board converse-stt` to also show the Converse-STT 1.0 figures next to the 2.0 runs.
6. **Export.** `export` writes the same runs as the JSON the [leaderboard website](https://www.useocular.com/converse/stt)
   consumes: `export/converse-stt/v2/board.json` (every run with pooled counts per conversation and track, plus the
   operating points in `runs-extra/`), one `transcripts/<run>.json` per board run (reference, normalized and raw
   hypothesis, word alignment) and the vendor marks under `logos/`. Models marked `prerelease = true` in `models.toml`
   are left out unless you pass `--include-prerelease`; `--generated 'Oct 21, 2026'` sets the build-date label.

### Publishing: two exports, two sites

Ocular publishes the board twice from the same runs: the public leaderboard on useocular.com and an internal
site that also shows models scored on pre-release deployments. The export is the only hand-off; neither site
reads a run directory. Release status is decided here, once, and never on a site:

```sh
uv run ocular-evals export --out export/public                          # what the public site gets
uv run ocular-evals export --out export/internal --include-prerelease   # the internal site: every run
```

- A model that must not be named in this repository yet goes in `models.local.toml` (git-ignored, merged over
  `models.toml`) with `prerelease = true`. Its run is dropped from the public export together with its
  transcript file, and `prerelease: true` is written on its row in the internal export so that site can mark it
  "Preview". When the model ships, move the table into `models.toml`, drop the flag, re-export, re-sync.
- `board.json` records `harness.version` and `harness.commit` (the git SHA of this checkout), so every published
  board names the exact scoring code. A change to the scorer, normalizer or dataset makes a new board **version**
  under the version policy, never an update to a published one; the site syncs refuse to overwrite a version whose
  harness version differs unless told it is a new version.
- `scripts/publish.sh` runs both exports, syncs each into its site checkout and opens the two review PRs; see
  the script header for the checkout paths it expects.

## Model configs

`models.toml` is the short form described above. Under the hood every entry becomes a full model config;
built-in examples live in `src/ocular_evals/models/*.toml`, and any TOML file with that shape works with
`--model path/to/file.toml`:

```toml
name = "gpt-4o-mini-transcribe"
display_name = "GPT-4o mini Transcribe"
vendor = "OpenAI"
provider = "openai-compatible"   # openai-compatible | openai-realtime | xai | meta | reson8 | omi | faster-whisper | nemotron-asr | cohere-transcribe
model = "gpt-4o-mini-transcribe"
env = ["OPENAI_API_KEY"]
href = "https://platform.openai.com/docs/guides/speech-to-text"
# group = "converse-stt-1.0"    # optional: put the run on the leaderboard's "Version 1 models" tab instead of the 2.0 board
# prerelease = true             # optional: a pre-release deployment; `export` skips the run unless --include-prerelease
#                               # (put such a table in the git-ignored models.local.toml if the model cannot be named here yet)

[options]
response_format = "json"
upload = "mp3"                   # OpenAI caps uploads at 25 MB
```

| provider            | what it calls                                                            | key options                                                                 |
| ------------------- | ------------------------------------------------------------------------ | --------------------------------------------------------------------------- |
| `openai-compatible` | `POST {base_url}/audio/transcriptions` (OpenAI and any compatible server) | `base_url`, `api_key_env`, `response_format`, `upload` (`flac`/`wav16k`/`mp3`), `chunking_strategy` (OpenAI server-side VAD chunking, needed for the gpt-4o-transcribe models on long audio), `timestamp_granularities` (Mistral: `"segment"` makes Voxtral decode per segment and avoids its repetition loop on long tracks), `chunk_s` + `search_s` (client-side: cut the track at a pause into chunks of at most `chunk_s` seconds and join the transcripts, for endpoints that cap request duration or size) |
| `openai-realtime`   | OpenAI Realtime API transcription session over WebSocket (`wss://api.openai.com/v1/realtime?intent=transcription`): 24 kHz PCM streamed at `speed`× real time, server VAD cuts turns, turns are joined in order. This is how Converse-STT 1.0 ran "GPT Realtime Whisper". | `api_key_env`, `speed` (default 10), `silence_ms` (VAD silence, default 500), `vad_threshold`, `prefix_padding_ms`, `prompt`, `language_hint` |
| `xai`               | `POST https://api.x.ai/v1/stt` (Grok Voice Transcribe)                   | `api_key_env`, `upload`, `format` (inverse text normalization), `filler_words` (default on: references are verbatim) |
| `reson8`            | `POST https://api.reson8.dev/v1/speech-to-text/prerecorded` (Reson8); the whole file is the request body, settings are query parameters, no model id | `api_key_env`, `upload`, `filler_mode` (`verbatim` default: references are verbatim; `natural`, `clean`), `include_words`, `custom_model_id` |
| `meta`              | `POST https://api.meta.ai/v1/asr/transcribe` (Muse Voice Transcribe); WAV only, 10 min per clip, so tracks are cut at a pause into chunks and the transcripts joined | `api_key_env`, `mode` (`ENDPOINTING`/`PUSH_TO_TALK`), `chunk_s` (≤ 600), `search_s` |
| `omi`               | asynchronous job API for long audio: create job, upload, poll for result | `base_url`, `api_key_env`, `response_format`, `poll_wait_s`, `deadline_s`   |
| `faster-whisper`    | local CTranslate2 Whisper                                                 | `device`, `compute_type`, `beam_size`, `vad_filter`, `condition_on_previous_text` |
| `nemotron-asr`      | local Transformers inference for NVIDIA's cache-aware streaming models (`nvidia/nemotron-speech-streaming-en-0.6b`, `nvidia/nemotron-3.5-asr-streaming-0.6b`); tracks are cut at pauses into ≤ 400 s pieces (the model's positional limit) | `lookahead_frames` (right context in 80 ms frames: 0 = 80 ms latency … 13 = 1.12 s, the most accurate; offline output at a lookahead equals streaming at that chunk size), `language` (locale prompt for the multilingual 3.5 model), `device` (`auto`/`cpu`/`mps`/`cuda`), `max_chunk_s`, `search_s` |
| `cohere-transcribe` | local Transformers inference for Cohere's open-weight `CohereLabs/cohere-transcribe-03-2026` (gated: accept the licence on Hugging Face and log in once); tracks are cut with the model's own energy-based splitter into ≤ 35 s pieces, transcribed in batches and joined | `device` (`auto`/`cpu`/`mps`/`cuda`), `dtype` (`auto` = bfloat16 on GPU, float32 on CPU), `batch_size`, `max_new_tokens`, `punctuation`, `language`, `skip_silence_dbfs` (optional peak-level gate that skips near-silent pieces instead of sending them to the decoder; off by default) |

Options never contain credentials; keys are read from the environment variables listed in `env`.
The `xai`, `meta` and `reson8` adapters need only `provider`, `key` and `model` in `models.toml`; their base URLs are built in.

## Layout

```
models.toml       the models this harness runs (url + key per model)
models.local.toml private overlay: pre-release or private models, merged over models.toml (gitignored)
scripts/publish.sh  export both profiles, sync both site checkouts, open the review PRs
src/ocular_evals/
  dataset.py      Hugging Face / local AudioFolder -> manifest.json
  text.py         normalization profiles
  metrics.py      EditCounts, WER / CER alignment
  config.py       model TOML loading
  registry.py     models.toml parsing / `models add`
  providers/      openai_compatible, openai_realtime, xai, meta, reson8, omi, faster_whisper, nemotron_asr, cohere_transcribe (+ retrying HTTP, audio prep / chunking)
  runner.py       resumable transcription + scoring of a run directory
  report.py       leaderboard HTML
  export.py       board + transcripts JSON for the website (`export`)
  models/         built-in model configs
  boards/         benchmark slug / version, conversation topics, open-weight and streaming lists, Converse-STT 1.0 figures
  templates/      leaderboard.html, vendor logos
docs/             technical report (Markdown + generated HTML) and its build script
tests/            pytest, no network
data/             datasets (gitignored)
runs/             run directories (gitignored)
runs-extra/       operating points: the same models at other settings, exported but not ranked (gitignored)
reports/          generated HTML (gitignored)
export/           website JSON (gitignored)
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
  (streaming / WebSocket vendors, batch job APIs) beyond the current `openai-compatible`, `openai-realtime`,
  `xai`, `meta`, `reson8`, `omi`, `faster-whisper`, `nemotron-asr` and `cohere-transcribe` backends.
- **More datasets and languages.** Run the same harness over additional Ocular AI conversation sets as they
  are published, including non-English ones.
- **Latency and cost.** Record time-to-transcript and per-minute pricing next to accuracy so the board can
  rank on more than error rate.
- **Published board sync.** Push verified harness results straight to the public
  [Converse-STT](https://www.useocular.com/converse/stt) leaderboard.
- **More of the Converse family.** Reuse the dataset manifest and report for further Converse benchmarks
  as they are announced, so the family shares one harness.

## License

Copyright (c) 2026 Ocular AI. This project is released under the [MIT License](LICENSE); you are free to
use, modify and redistribute it, including commercially, as long as the copyright and permission notice are
kept. Contributions are accepted under the same license.

The harness code is MIT; the Ocular AI conversation datasets it downloads from Hugging Face carry their own
terms, stated on each dataset card. Model outputs and scores you generate belong to you.

## Contact

Questions about the benchmark, a model you would like on the board, or the Hi-Fi conversational datasets:
[research@useocular.com](mailto:research@useocular.com). Bugs and contributions to the harness are welcome as
GitHub issues and pull requests.
