# Converse-STT 2.0: Benchmarking Speech-to-Text on Real Two-Person American English Conversation

**Ocular AI · Technical report · October 2026 · harness `ocular-evals` 0.1.0**

Converse-STT 2.0 measures how accurately speech-to-text systems transcribe unscripted, two-person American English conversation when each speaker is recorded on their own microphone for a full 15-minute session. This report documents the benchmark end to end: how the recordings were collected and transcribed, how the audio reaches each model, how text is normalized, how word and character error rates are computed and pooled, what the 20 systems on the board scored, and what the errors actually are.

| | |
| --- | --- |
| Conversations / speaker tracks | 4 / 8 |
| Conversation timeline | 60.4 min (120.6 min of per-speaker audio) |
| Reference words scored | 12,555 (47,784 characters) |
| Overlapping speech | 15.4 % of all speech time |
| Systems on the board | 20 (each vendor's latest model, every reachable Converse-STT 1.0 model, and three open-weight models run locally; 15 proprietary, 5 open-weight) |
| Primary metric | Pooled WER, Whisper English normalization |
| Best / median / worst WER | 1.22 % / 4.32 % / 13.82 % |

---

## 1. Overview

Public ASR leaderboards are dominated by short, single-speaker clips. Conversation is different: speakers overlap, backchannel ("yeah", "right", "mm-hmm"), restart words mid-syllable, and talk for a quarter of an hour without a break. Converse-STT 1.0, scored in September 2026 with Cekura, showed that twelve of fifteen models had a higher error rate on Ocular AI's conversational recordings than on public Pipecat clips and that the leading model changed with the dataset.

Converse-STT 2.0 keeps the same four recordings and makes the evaluation reproducible by anyone. Every number on the 2.0 board is produced by the open-source harness in this repository ([MIT](../LICENSE)): the dataset manifest, the per-model inference configuration, the normalizer, the scorer and the report are all code, and every run directory keeps the raw transcripts and the normalized texts that produced its score.

The design follows established conventions for conversational ASR evaluation (separate channel per speaker, verbatim references, Whisper English normalization, summed edit counts), so that a model's number here can be read next to numbers from other public harnesses that use the same conventions. The implementation is independent and deliberately small: one Python package, one manifest format, one command per step.

---

## 2. What is in the benchmark

### 2.1 Collection protocol

The audio is the public four-conversation sample of Ocular AI's American English Hi-Fi full-duplex collection, published on Hugging Face as [`OcularAIInc/AMERICAN-ENGLISH-TRANSCRIBED-HIFI-FULL-DUPLEX-TWO-SPEAKER-CONVERSATIONAL-DATASET-SAMPLE`](https://huggingface.co/datasets/OcularAIInc/AMERICAN-ENGLISH-TRANSCRIBED-HIFI-FULL-DUPLEX-TWO-SPEAKER-CONVERSATIONAL-DATASET-SAMPLE) under CC BY 4.0. The harness pins revision `29b74ad449eb99a1f9937dc0f75d35333ca866d5`.

- **Setting.** Two participants hold a conversation on Ocular's recording platform. Each participant is captured on a separate microphone into a separate file for the entire session (full duplex), so there is no mixing and no voice-activity gating between speakers at the recording stage.
- **Prompting.** Each session opens with a supplied discussion prompt that one participant reads or paraphrases (for example, "Consider the ethical and practical implications of using drones in everyday life…"). The prompt is part of the recording and of the reference; everything after it is unscripted.
- **Participants.** Eight distinct speakers, all self-reporting American English as accent and English as native language; five female, three male; self-reported age bands 35–45 (1), 45–55 (3) and not reported (4). Demographics are self-reported and were not verified.
- **Sessions.** Recorded 12 and 15 August 2026 (UTC). Topics: drones and privacy; interpreting news sources; brand loyalty and customer engagement; fitness plans.
- **Synchronization.** The two files of a session do not start at the same instant. Each filename carries the capture start in epoch milliseconds (`<room_start_ms>-<speaker_id>-hifi-audio-<capture_start_ms>.flac`); the later track in each session starts 1.298–2.125 s after the earlier one. The harness derives this offset from the filenames and shifts the later track's word timestamps onto a shared conversation timeline (Section 2.6).

### 2.2 Audio specification

All eight files are mono 24-bit PCM FLAC at 48 kHz, delivered as recorded (no resampling, trimming or loudness processing). The harness never alters the archived audio; conversions for upload are made from it per request (Section 3.2).

| Track | Speaker | Duration | Size | RMS | Peak | Noise floor (5th pct of 100 ms RMS) |
| --- | --- | --- | --- | --- | --- | --- |
| 0fcd38f2 / ch1 | M | 904.3 s | 42.6 MB | −29.8 dBFS | −4.5 dBFS | digital silence (gated) |
| 0fcd38f2 / ch2 | F | 901.3 s | 67.8 MB | −29.7 dBFS | −3.1 dBFS | −92.5 dBFS |
| 2bb0cfe2 / ch1 | F, 45–55 | 905.1 s | 67.6 MB | −28.2 dBFS | 0.0 dBFS | −82.4 dBFS |
| 2bb0cfe2 / ch2 | M | 904.8 s | 30.1 MB | −29.5 dBFS | 0.0 dBFS | −78.7 dBFS |
| 86c27418 / ch1 | F, 45–55 | 904.9 s | 65.0 MB | −29.8 dBFS | −3.0 dBFS | −91.6 dBFS |
| 86c27418 / ch2 | F | 904.8 s | 64.7 MB | −27.9 dBFS | −0.1 dBFS | −89.8 dBFS |
| ea217639 / ch1 | F, 35–45 | 907.6 s | 47.8 MB | −27.6 dBFS | −2.5 dBFS | −85.8 dBFS |
| ea217639 / ch2 | M, 45–55 | 905.8 s | 68.6 MB | −29.2 dBFS | −3.7 dBFS | −69.5 dBFS |

Three observations matter for interpreting results. First, recording levels are consistent (−28 to −30 dBFS RMS) and the floors are very low: this is close-talk, quiet-room audio, not telephone or far-field speech. Second, two tracks in conversation 2bb0cfe2 reach 0.0 dBFS, so some clipping on loud syllables is possible. Third, track 0fcd38f2/ch1 has a noise floor of true digital zero, which indicates client-side gating or noise suppression on that participant's input chain; when that gate is open, the partner's voice is audible on the channel. This single track accounts for essentially all of the cross-talk insertions in the results (Section 5.4).

### 2.3 Transcription protocol and reference conventions

According to the collection's documentation, references were produced by an ASR-assisted first pass followed by human transcription correction, a second human review, and an Ocular audit. Word-level timestamps were produced afterwards by forced alignment of the reviewed text to the audio; non-speech event times are interpolated between neighbouring aligned words rather than measured independently.

References are **verbatim**: they keep fillers, repetitions, false starts and word fragments, and annotate non-speech events with SGML-style tags.

| Convention | Example from the references | Count in the 8 tracks |
| --- | --- | --- |
| Audible breath | `<breath/>` | 616 |
| Standalone laughter | `<laugh/>` | 108 |
| Laughter spanning speech | `<laugh>…</laugh>` | 7 pairs |
| Speech present, not transcribable | `<spn/>` | 20 |
| Unclear audio | `<unclear/>` | 4 |
| Throat clearing | `<clear-throat/>`, `<Clear-throat/>` | 7 |
| Word fragment / restart | "I- I'll listen", "Con- consider" | 202 |
| Immediate repetition | "the the", "I I" (after normalization) | 220 |
| Trailing-off | "…" | 86 |
| Backchannel spelled out | "Mm-hmm", "Uh-huh" | 62, 3 |
| Fillers | "um", "uh" | removed by the normalizer (Section 3.3) |
| Spelled letter sequences | "A-I", "G-P-S", "N-P-R" | 6 + 3 + 2 |
| Numbers | written as words: "twenty different cameras", "a thousand dollars" | no digits in any reference |
| Contractions | "it's" 116, "I'm" 80, "don't" 74, "that's" 63, "there's" 43 … | ≈ 700 |

The references contain no digits at all; every number is spelled out as spoken. Contractions are written as spoken. The Hugging Face card notes known imperfections: two segments without word arrays, one segment ending 69 ms after its track, and alignment that is less reliable for short fillers. None of these affect text scoring, which uses the full-track transcript, not the word arrays.

### 2.4 Dataset statistics

Speech time below is computed from the word-level timestamps after merging same-speaker words separated by ≤ 200 ms (a pause-merging threshold commonly used for multi-speaker conversational corpora). Overlap is the time both speakers are active, as a share of total speech time. Turns are same-speaker stretches separated by more than one second of that speaker's silence.

| Conversation | Topic | ch1 | ch2 | Timeline | Ref words (ch1 + ch2) | Speech (ch1 / ch2) | Overlap | Silence | Turns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0fcd38f2 | Drones, privacy | M | F | 904 s | 1,257 + 1,806 = 3,063 | 331 s / 569 s | 115 s (12.8 %) | 120 s (13 %) | 69 / 42 |
| 2bb0cfe2 | News sources | F 45–55 | M | 906 s | 1,458 + 1,841 = 3,299 | 483 s / 512 s | 193 s (19.4 %) | 104 s (11 %) | 59 / 61 |
| 86c27418 | Fitness plans | F 45–55 | F | 906 s | 1,655 + 1,547 = 3,202 | 480 s / 443 s | 131 s (14.2 %) | 114 s (13 %) | 35 / 67 |
| ea217639 | Brand loyalty | F 35–45 | M 45–55 | 908 s | 1,447 + 1,544 = 2,991 | 509 s / 472 s | 147 s (14.9 %) | 74 s (8 %) | 25 / 40 |
| **Total** | | | | **3,624 s** | **12,555** | **3,798 s** | **585 s (15.4 %)** | **412 s (11 %)** | **398** |

For scale: published two-speaker conversational corpora typically report overlap in the 3–15 % range of speech time; Converse-STT sits at the high end of that range in every conversation, and conversation 2bb0cfe2 (19.4 %) is above it.

Lexically the corpus is small-vocabulary spontaneous speech: 1,551 distinct normalized word types over 12,555 tokens. The most frequent content is conversational scaffolding: "like" is the third most frequent word (383 tokens, 3.1 % of the corpus), "know" 303 (almost all in "you know"), "yeah" 186, "right" 119, "just" 125. A system that cleans disfluencies or drops backchannels is penalized in proportion, which is intended: the benchmark asks for what was said.

### 2.5 Conversational phenomena

The references preserve the behaviour that makes conversation hard. Raw excerpts (markup left in):

**Fillers, backchannels, and a self-interruption** (0fcd38f2/ch2, F):

> Um, I think the most beneficial it could be is for deliveries, absolutely. Getting my stuff quicker. Love it. Um, `<laugh/>` other than that, as far as, like, surveillance, `<breath/>` uh, `<spn/>` I'm thinking specifically with a lady brain that, no. `<breath/>` there's too many creeps already. `<laugh/>`

**Restarts and fragments** (2bb0cfe2/ch2, M):

> So the strategy I use is I got these, uh, `<breath/>` noise-canceling headphones, right? And I put them on, and I- I'll listen to, like, the Rocky soundtrack, `<breath/>` and I'll just, um, I'll walk down the street, and I'll go to- um, I'll go to, like, this auto body shop…

**A prompt read aloud, then breath-tagged spontaneous speech** (ea217639/ch1, F):

> Consider a business that wants to enhance its brand loyalty `<breath/>`. What strategies would you suggest to deep-in customer engagement `<breath/>` and foster long-term relationships? Mm-hmm. Yes `<laugh/>` . `<breath/>` You know, I think you covered some great ones already, like the Cracker Jack box…

**Numbers as spoken** (various): "on twenty different cameras", "three hundred and sixty", "a check for a thousand dollars", "a solid block of thirty, sixty minutes", "five thousand dollars or … a fifty doll-".

### 2.6 From the published files to a scoring manifest

`ocular-evals dataset pull` downloads `train/metadata.jsonl` and the eight FLACs from the pinned revision and writes `data/converse-en-sample/manifest.json`, the only dataset artefact the rest of the harness reads. For each session it:

1. groups the two rows by `session_id`;
2. parses the capture start from each filename, orders tracks by capture time (then `speaker_id`), names them `ch1`, `ch2`, and records `offset_s = (capture_ms − min capture_ms) / 1000` (0.000 / 2.125, 0.000 / 1.321, 0.000 / 1.298, 0.000 / 2.109 s);
3. copies each FLAC byte-for-byte to `audio/<session>/<chN>.flac` and records its SHA-256, duration (from the container header) and sample rate;
4. stores the full `transcript` string as the reference, and the `segments[].words` entries with `start`/`end` shifted by `offset_s` as word timings on the shared timeline;
5. keeps the self-reported metadata fields (`accent_self_reported`, `age_range`, `gender_self_reported`, `native_language`, `take_id`, `datetime_utc`).

The manifest records the Hugging Face repo, revision and the SHA-256 of `metadata.jsonl`. The dataset's identity for the rest of the pipeline is a 16-hex-character digest of the sorted `(conversation, track, sha256)` triples; a run refuses to resume or re-score if that digest changes, and every track's checksum is re-verified immediately before it is uploaded. `ocular-evals dataset verify` re-checks all files against the manifest.

---

## 3. Evaluation methodology

### 3.1 Task definition

The task is **batch, single-channel, single-speaker transcription of a 15-minute track** with a language hint of `en`. Each of the eight tracks is sent to the system whole (or in the largest pieces the system's API allows, Section 3.2) and the returned text is scored against that track's reference. There is no mixing of the two speakers, no diarization requirement, and no streaming; the score is about what the system writes down for one person talking in a real conversation, with the other person's voice occasionally audible through the microphone.

Scoring per separate channel is the standard "separated channels" condition for dual-microphone conversational corpora. A mixed-channel, speaker-attributed condition (cpWER / DER) is on the harness roadmap and is not part of this report.

### 3.2 Inference harness

`ocular-evals run --model <name>` resolves the entry in `models.toml` to a provider adapter and processes the selected tracks sequentially. An entry may carry `group = "converse-stt-1.0"`, a display field (it does not enter the model digest) that keeps a run off the 2.0 board and shows it on the leaderboard's Version 1 tab only; no current entry uses it, because every re-run 1.0 model is also ranked on the 2.0 board (Section 5.7).

**Provider adapters.**

| Adapter | Protocol | Used for |
| --- | --- | --- |
| `openai-compatible` | `POST {base_url}/audio/transcriptions`, multipart (`file`, `model`, `language`, `response_format=json`, optional `chunking_strategy`, `timestamp_granularities`) | OpenAI (GPT Transcribe, GPT-4o Transcribe, GPT-4o mini Transcribe), Mistral, and every model routed through OpenRouter |
| `xai` | `POST https://api.x.ai/v1/stt`, multipart (`model`, `language`, `format`, `filler_words`, `file`) | Grok Voice Transcribe 2.0 |
| `reson8` | `POST https://api.reson8.dev/v1/speech-to-text/prerecorded`, raw audio body (`application/octet-stream`, container auto-detected), settings as query parameters (`language`, `filler_mode=verbatim`, `include_words`, `include_timestamps`), `Authorization: ApiKey` | Reson8 |
| `meta` | `POST https://api.meta.ai/v1/asr/transcribe`, multipart JSON `request` + mono 16 kHz PCM WAV `audio`; ≤ 600 s and ≤ 32 MB per request | Muse Voice Transcribe 1.0 |
| `omi` | asynchronous job API: create job, upload, poll for result | Omi General, Omi Medical |
| `faster-whisper` | local CTranslate2 inference | available; not used for any number in this report |
| `nemotron-asr` | local 🤗 Transformers inference (`AutoModelForRNNT`) on this machine's GPU/CPU; mono 16 kHz float input, no upload | Nemotron 3 ASR (EN), Nemotron 3.5 ASR |
| `cohere-transcribe` | local 🤗 Transformers inference (`CohereAsrForConditionalGeneration`, greedy, `max_new_tokens=256`); the model's own feature extractor cuts the track into ≤ 35 s pieces at the quietest point of each window, the pieces are decoded in batches of 4 and joined with a space; optional peak-level gate (`skip_silence_dbfs`) skips near-silent pieces | Cohere Transcribe |
| `openai-realtime` | WebSocket transcription session on the OpenAI Realtime API: 24 kHz PCM16 appended at 10× real time, server VAD cuts turns, each turn's `input_audio_transcription.completed` event is collected and the turns joined in order | GPT Realtime Whisper |

**Upload preparation.** The archived FLACs are 30–69 MB and 48 kHz / 24-bit, which exceeds or stresses most upload paths (OpenAI and OpenRouter cap at 25 MB; several vendors' TLS front-ends failed mid-upload with `SSL bad record mac` above roughly 14 MB during our runs). Each model entry therefore declares an `upload` format, produced on the fly from the archived file and discarded afterwards:

- `mp3`: ffmpeg 8.1, mono, 16 kHz, 64 kbps CBR, LAME defaults (≈ 7 MB per track). Used for every `openai-compatible` and `xai` entry.
- `wav16k`: mono 16 kHz 16-bit PCM, resampled in-process with linear interpolation from the 48 kHz source (≈ 29 MB per track). Used only where the vendor rejects compressed audio (AssemblyAI via OpenRouter, Meta).
- `flac`: the original file, untouched.

Downsampling to 16 kHz discards content above 8 kHz. Published ASR front-ends (Whisper, Conformer/Parakeet, Voxtral) operate at 16 kHz, and we have no evidence that any evaluated API uses more, so this is not expected to change results; but it is a choice, and it is recorded in every run's `run.json`. The one direct comparison we have points the other way for robustness rather than accuracy: Voxtral's repetition loop (Section 5.6) occurred on the MP3 and not on the original FLAC.

**Client-side chunking.** Some endpoints cap request duration or size below a 15-minute track: AssemblyAI via OpenRouter (≈ 4 MB WAV), Meta (600 s), the Nemotron models' Transformers implementation (5,000 encoder frames = 400 s per request) and Google's Chirp 3 synchronous API (60 s). For these, the harness cuts the track into pieces of at most `chunk_s` seconds. Cut points start evenly spaced and are then moved to the quietest 0.5 s window within ± `search_s` seconds of the even split (minimum of the moving sum of squared 16 kHz samples), clamped so that no chunk can exceed `chunk_s`. The chunks are transcribed in order, the texts are joined with a single space, and segment timestamps are offset by the chunk start. The cut algorithm, chunk boundaries and per-chunk byte counts are stored in the hypothesis's `raw` field. Chunking introduces a risk of losing or duplicating a word at a boundary; the pause-seeking cut keeps that small but not zero (Section 5.6).

**Server-side chunking.** OpenAI's GPT-4o-family transcription models silently drop large portions of a 15-minute file unless the request sets `chunking_strategy=auto` (the diarize variant refuses without it). The harness supports the parameter for any OpenAI-protocol model; the GPT-4o Transcribe and GPT-4o mini Transcribe board rows use it, and GPT Transcribe does not need it. Section 5.6 quantifies the difference.

**Reliability.** Transport errors, timeouts and HTTP 408/409/425/429/500/502/503/504 are retried up to six times with exponential backoff (1 s base, 30 s cap, 25 % jitter), honouring `Retry-After`. A 429 whose body reports exhausted quota or credits is not retried. Every failed attempt is appended to `failures.jsonl` with the error message redacted of any secret; after three consecutive failures the run stops. Runs are resumable: a track is skipped if `hypotheses.jsonl` already holds a row for the same model digest and the same audio SHA-256. The model digest is a hash of the provider, model id and every option that can change the output (not of timeouts or URLs), so changing `chunking_strategy` or `timestamp_granularities` forces a fresh run. A run directory is locked with `flock` while in use.

**What is recorded per track.** Raw returned text, detected language and duration where the API reports them, segments, the full raw response body (minus credentials), wall-clock latency from request start to parsed response, and the audio checksum.

### 3.3 Text normalization

Both the reference and the hypothesis pass through the same function before any comparison. The profile used for every number in this report is `whisper`:

| Step | Action | Detail |
| --- | --- | --- |
| 1 | Unicode NFKC | Compatibility characters folded (full-width forms, ligatures). |
| 2 | Strip transcript markup | SGML tags `<…>` / `</…>`, bracketed notes `[…]`, and parenthesised asides `(…)` (nested) are replaced by a space. This removes every event tag in Table 2.3 and, on the hypothesis side, any `[laughter]`, `(inaudible)` style annotations a model emits. |
| 3 | Fold apostrophes | `’ ‘ ʼ ´ \`` → `'`. |
| 4 | Whisper English normalizer | `whisper-normalizer` 0.1.12, `EnglishTextNormalizer`, the reference implementation from OpenAI's Whisper paper. Lowercases; **removes the fillers `hmm`, `mm`, `mhm`, `mmm`, `uh`, `um`**; expands contractions (`i'm → i am`, `won't → will not`, `gonna → going to`, `y'all → you all`); maps British to American spellings; expands abbreviations (`dr. → doctor`); converts spoken numbers to digits including currency and percent (`twenty → 20`, `a thousand dollars → $1000`, `a hundred percent → 100%`); replaces remaining punctuation (except inside numbers) with spaces. |
| 5 | Drop non-lexical tokens | Any token with no letter or digit (a stray `.` left after a filler was removed) is dropped. |
| 6 | Collapse whitespace, lowercase | |

Worked example, reference 0fcd38f2/ch2:

```
RAW   Um, I think the most beneficial it could be is for deliveries, absolutely. … Um, <laugh/> other
      than that, as far as, like, surveillance, <breath/> uh, <spn/> I'm thinking specifically with a
      lady brain that, no. <breath/> there's too many creeps already.
NORM  i think the most beneficial it could be is for deliveries absolutely … other than that as far as
      like surveillance i am thinking specifically with a lady brain that no there is too many creeps
      already
```

Consequences a reader should know, all of which apply identically to both sides:

- **`um`/`uh`/`hmm`/`mm-hmm` are not scored.** "Mm-hmm" becomes `mm hmm` and both tokens are removed. A system is neither rewarded nor penalized for transcribing these particular fillers. "like", "you know", "yeah", "right", "okay", "oh", repetitions and fragments **are** scored.
- The filler list is Whisper's and is not exhaustive. Spellings outside it survive normalization: Deepgram's `mhmm`, Omi's and Qwen's `uh-huh → huh`, Qwen's `嗯`. Since the reference form was deleted, these count as insertions (Section 5.4).
- Contraction expansion means normalized word counts can exceed raw counts (12,326 raw tokens after markup removal become 12,555 scored words). The denominator of every WER in this report is the normalized reference length.
- Number conversion is applied to spoken forms on both sides, so "twenty different cameras" and "20 different cameras" agree. It has edge cases: "oh" is mapped to `0` (75 reference tokens), and "thirty, sixty minutes" becomes `3060 minutes`. These are symmetric and rarely create errors, but they make the normalized text look odd.
- Hyphens become spaces. The references spell acronyms as letters ("A-I", "G-P-S", "N-P-R") and write "push-ups"; after normalization these are `a i`, `g p s`, `push ups`, while every model writes `ai`, `gps`, `pushups`. This costs each model a fixed 10–15 errors (≈ 0.1 WER points) and is the single systematic reference-convention penalty we have identified. It is left in place rather than special-cased because it is applied uniformly and because fixing it would be a change to the published normalizer.

A second profile, `plain` (lowercase, strip punctuation except apostrophes and hyphens within words, keep fillers), is available with `--profile plain` for inspection; it is not used on the board.

### 3.4 Scoring

For each track the normalized reference and hypothesis are tokenized two ways: **words** = whitespace-separated tokens; **characters** = every non-whitespace character. Each pair of token sequences is aligned with `jiwer` 4.0 (`process_words` with identity transforms, i.e. a standard Levenshtein alignment with unit cost for substitution, deletion and insertion; `rapidfuzz` 3.14 backend). The result is stored as an `EditCounts` record: hits H, substitutions S, deletions D, insertions I.

Rates are derived, never stored:

\[
\mathrm{WER} = \frac{S + D + I}{H + S + D}, \qquad
\mathrm{CER} = \frac{S_c + D_c + I_c}{H_c + S_c + D_c}
\]

**Pooling.** Counts are summed across the two tracks of a conversation and across the four conversations; the headline WER is the ratio of the summed counts (total errors / total reference words). The harness never averages percentages, so a short track cannot carry the same weight as a long one and a catastrophic track (a decoder loop) contributes exactly its error count. WER is unbounded above; the pre-fix Voxtral run scored 107 % because it produced 12,800 inserted words on one track.

A track with no hypothesis (failed after retries) is excluded from the sums and the run is marked incomplete; every run on the board is complete (8/8 tracks). A hypothesis that is empty text scores as all deletions.

CER is reported alongside WER as a tiebreaker and as a check that WER is not being driven by tokenization (e.g. compounding `pushups`); the primary metric for English is WER.

**Secondary measurement.** Wall-clock latency per track is recorded and summarized as the median across the eight tracks. It is an end-to-end batch number including upload and any proxy hop and is not comparable across routes; it is reported for context, not ranked.

### 3.5 Reproducibility and audit trail

Each run directory `runs/<model>/` contains:

| File | Content |
| --- | --- |
| `run.json` | Harness version, full resolved model config, model digest, dataset root + digest + Hugging Face source, normalization profile, timestamps |
| `hypotheses.jsonl` | One row per transcribed track: raw text, segments, language, latency, audio SHA-256, model digest, full raw API response |
| `failures.jsonl` | One row per failed attempt: status, retryable flag, redacted message |
| `scores.jsonl` | Per-track rows with WER/CER counts **and the exact normalized reference and hypothesis strings that were aligned**; per-conversation rows |
| `summary.json` | Pooled counts and rates, per-conversation breakdown |

`ocular-evals report` renders every `summary.json` into a single self-contained HTML leaderboard with WER/CER toggle, per-conversation drill-down, and reference/hypothesis side-by-side for every track; segments repeated many times are flagged as a probable decode loop. The complete command sequence is in Appendix C.

### 3.6 Streaming systems

Converse-STT scores a finished transcript of a finished recording; it is a batch (offline) benchmark and does not measure latency, partial-result stability or endpointing. Several systems on the board also exist as streaming products (Deepgram, AssemblyAI, ElevenLabs, xAI and Reson8 all sell real-time endpoints), and for those the board uses the vendor's batch endpoint, as a vendor evaluation would. Two systems, NVIDIA's Nemotron 3 ASR and Nemotron 3.5 ASR, exist *only* as streaming models: cache-aware FastConformer-RNNT encoders in which the right attention context ("lookahead", in 80 ms frames) fixes how much future audio each frame may see. Because the streaming steps never overlap and the cache reproduces the full left context, offline inference at a given lookahead yields the same transcript the streaming path produces at that chunk size; the lookahead is therefore the single knob that moves these models along their latency–accuracy curve (80 ms, 160 ms, 320 ms, 560 ms, 1.12 s).

The policy is: a streaming-only model's **board row** is its highest-accuracy operating point (lookahead 13 frames = 1.12 s), which is the configuration a batch deployment would use and the one comparable to the other systems' batch endpoints. The row's `notes` field names the setting. Lower-latency operating points are **not ranked**; they are reported as a separate table (Section 5.8) so that the accuracy cost of latency is visible without putting a 80 ms streaming configuration in the same ordering as a batch API. The same rule will apply to any future streaming-capable system that has no batch endpoint: one board row at its most accurate setting, labelled, with the latency sweep alongside. One streaming system is scored through its streaming path on purpose: Converse-STT 1.0's "GPT Realtime Whisper" is the OpenAI Realtime API's transcription session (server VAD, `whisper-1`), a product that exists only as a stream, and reproducing the 1.0 row means running it that way (Section 5.7). Its board row is labelled accordingly, and its score should be read as the accuracy of that product on this audio, not of `whisper-1` as a batch decoder.

---

## 4. Evaluated systems and inference configuration

**Board policy.** The leaderboard ranks two sets of systems. First, each vendor's latest general-purpose transcription model: where a vendor offers several current models we take the newest generation (GPT Transcribe; Gemini 3.5 Transcribe; Grok Voice Transcribe 2.0 rather than Grok STT 1.0; MAI-Transcribe 2 rather than 1.5), the flagship size within a generation (Qwen3-ASR 1.7B rather than 0.6B), and the general-domain model rather than a domain variant (Omi General rather than Omi Medical); Cohere Transcribe is Cohere's only transcription model. Second, every model that was scored in Converse-STT 1.0 and that the credentials we hold can reach, so that the 2.0 board can be read against the 1.0 board model for model: this adds Chirp 3, GPT-4o Transcribe, GPT-4o mini Transcribe, GPT Realtime Whisper and Reson8 to the vendor-latest set (Nova-3 and Universal-3.5 Pro are in both sets; Reson8 is both the 1.0 leader and its vendor's only model). As a result Google has two rows and OpenAI four; within a vendor the rows are independent runs and are ranked independently. Sibling and older-generation models on neither list (Grok STT 1.0, MAI-Transcribe 1.5, Qwen3-ASR 0.6B, Omi Medical, GPT-4o Transcribe Diarize, Whisper large-v3) were run with the same harness on the same data; their numbers are in Appendix D and are referred to below where they illuminate a finding, but they are not ranked. NVIDIA has three rows for a different reason: Parakeet TDT 0.6B v3 is its hosted batch model, while Nemotron 3 ASR (English) and Nemotron 3.5 ASR (multilingual) are a separate, streaming-only family that NVIDIA positions as a complementary product rather than a successor (and recommends the English model for English transcription); both were added at the request of the board's owner and run locally at their batch-quality setting (Section 3.6).

Twenty systems are on the board. Fifteen hosted batch systems were run on 11 October 2026 between 02:49 and 10:09 UTC, the two local Nemotron models on 12 October 2026, and the GPT Realtime Whisper session, Reson8 and the local Cohere Transcribe on 13 October 2026 (per-run timestamps in Appendix A), all against the manifest built on 8 October 2026. Nine were called on the vendor's own endpoint with the vendor's key; three were run locally with open weights; eight through [OpenRouter](https://openrouter.ai)'s `/audio/transcriptions` gateway, which exposes a catalogue of transcription models behind one OpenAI-style API and one key. OpenRouter adds a proxy hop to latency, may route an id to more than one upstream host, and does not forward every vendor-specific parameter (notably `timestamp_granularities`); entries carry `notes = "via OpenRouter"` and the board says so.

| System | Vendor | Route / endpoint | API model id | Upload | Request parameters beyond defaults |
| --- | --- | --- | --- | --- | --- |
| Scribe v2 | ElevenLabs | OpenRouter | `elevenlabs/scribe-v2` | mp3 | — |
| Reson8 | Reson8 | direct `api.reson8.dev/v1/speech-to-text/prerecorded` | — (no model id; current pipeline) | mp3 | `filler_mode=verbatim`, `language=en`, `include_words`, `include_timestamps` |
| Grok Voice Transcribe 2.0 | xAI | direct `api.x.ai/v1/stt` | `grok-voice-transcribe-2.0` | mp3 | `format=true`, `filler_words=true`, `language=en` |
| Gemini 3.5 Transcribe | Google | OpenRouter | `google/gemini-3.5-transcribe` | mp3 | — |
| Muse Voice Transcribe 1.0 | Meta | direct `api.meta.ai/v1/asr/transcribe` | `muse-voice-transcribe-1.0` | wav 16 kHz PCM16 | `mode=ENDPOINTING`, client chunks ≤ 540 s cut at pauses (2 per track) |
| Nova-3 | Deepgram | OpenRouter | `deepgram/nova-3` | mp3 | — |
| Omi General | Omi | direct `api.omi.health` job API | `omi-general-1` | flac | — |
| Chirp 3 | Google | OpenRouter | `google/chirp-3` | mp3 | client chunks ≤ 55 s, search ± 5 s (17 per track; 60 s synchronous limit) |
| MAI-Transcribe 2 | Microsoft AI | OpenRouter | `microsoft/mai-transcribe-2` | mp3 | — |
| Nemotron 3 ASR Streaming (EN) 0.6B | NVIDIA (open weights) | local, 🤗 Transformers 5.19 / torch 2.14, Apple M4 Max (MPS) | `nvidia/nemotron-speech-streaming-en-0.6b` | 16 kHz float, in-process | `lookahead_frames=13` (1.12 s), client chunks ≤ 360 s cut at pauses (3 per track) |
| Parakeet TDT 0.6B v3 | NVIDIA (open weights) | OpenRouter | `nvidia/parakeet-tdt-0.6b-v3` | mp3 | — |
| GPT Transcribe | OpenAI | direct `api.openai.com/v1` | `gpt-transcribe` | mp3 | — |
| Qwen3-ASR 1.7B | Alibaba (open weights) | OpenRouter | `qwen/qwen3-asr-1.7b` | mp3 | — |
| GPT-4o mini Transcribe | OpenAI | direct `api.openai.com/v1` | `gpt-4o-mini-transcribe` | mp3 | `chunking_strategy=auto` (see 5.6) |
| Universal-3.5 Pro | AssemblyAI | OpenRouter | `assemblyai/universal-3-5-pro` | wav 16 kHz PCM16 | client chunks ≤ 120 s, search ± 10 s (8 per track; route rejects mp3 and > ≈ 4 MB) |
| Nemotron 3.5 ASR Streaming 0.6B | NVIDIA (open weights) | local, as above | `nvidia/nemotron-3.5-asr-streaming-0.6b` | 16 kHz float, in-process | `lookahead_frames=13` (1.12 s), `language=en-US` prompt, client chunks ≤ 360 s (3 per track) |
| Voxtral Mini Transcribe 2 | Mistral AI | direct `api.mistral.ai/v1` | `voxtral-mini-2602` | mp3 | `timestamp_granularities=segment` (see 5.6) |
| Cohere Transcribe | Cohere (open weights) | local, 🤗 Transformers 5.19 / torch 2.14, Apple M4 Max (MPS, bfloat16) | `CohereLabs/cohere-transcribe-03-2026` | 16 kHz float, in-process | `language=en`, `punctuation=true`, greedy `max_new_tokens=256`; the model's own ≤ 35 s pieces (28–29 per track), batches of 4; pieces peaking below −35 dBFS skipped (4 of 228; see 5.6) |
| GPT Realtime Whisper | OpenAI | direct `wss://api.openai.com/v1/realtime?intent=transcription` | `whisper-1` | 24 kHz PCM16 stream, fed at 10× real time | `turn_detection=server_vad` (`silence_duration_ms=500`), `language=en`; one request per VAD turn (see 5.7) |
| GPT-4o Transcribe | OpenAI | direct `api.openai.com/v1` | `gpt-4o-transcribe` | mp3 | `chunking_strategy=auto` (see 5.6) |

All requests send `language=en` (BCP-47 primary subtag of the conversation's `en-US`) and `response_format=json` where the protocol has it. No prompt, vocabulary boost, temperature or seed is sent unless listed; vendors' defaults apply. Grok's `filler_words=true` and `format=true` match the verbatim reference convention (the harness's normalizer removes `um`/`uh` anyway, so the setting is not decisive).

Systems considered for the board that could not be run with the credentials available, and why: Inworld, Cartesia, Smallest AI, Speechmatics and Gradium (no API access through our keys or OpenRouter; Section 5.7 lists what each would need); Grok Voice Transcribe 2.0 streaming (WebSocket-only; the batch endpoint is on the board); general-purpose chat models with audio input (not transcription endpoints, so not comparable); other open-weight ASR checkpoints that would need self-hosting (the Nemotron streaming models and Cohere Transcribe were self-hosted and are on the board; more can be added through the `nemotron-asr`, `cohere-transcribe` or `faster-whisper` adapters or a new local adapter). NVIDIA also serves the two Nemotron models through its hosted NIM at build.nvidia.com (Riva gRPC); that route needs an NVIDIA API key, which we did not hold, so the local weights were used instead. An OpenRouter route for Voxtral was run and removed from the board because the route cannot carry the parameter that prevents the model's repetition loop (Section 5.6).

**What OpenRouter does and does not carry.** OpenRouter's `/models` catalogue lists chat and audio-understanding models only; transcription models are not enumerated anywhere in the API, so the only way to learn whether `/audio/transcriptions` serves a given id is to send it a request. On 13 October 2026 we probed the endpoint with a one-second WAV for every plausible id of the vendors missing from the board (`speechmatics/linden`, `speechmatics/enhanced`, `speechmatics/standard`, `speechmatics/ursa-2`, `cartesia/ink-2`, `cartesia/ink-whisper`, `smallest/pulse`, `smallest-ai/pulse`, `inworld/stt-1`, `inworld/inworld-stt-1`, `gradium/gradium-stt`, `reson8/reson8`, `deepgram/flux-general-en`, `deepgram/flux`, `deepgram/nova-2`, `google/chirp-2`, `google/chirp`, `soniox/stt-async-v4`, `gladia/solaria-1`, `fireworks/whisper-v3-turbo`, `ibm/granite-speech-3.3-8b`). Every one returned `Model <id> does not exist` (HTTP 400); the two controls, `google/chirp-3` and `openai/whisper-1`, were accepted and billed. The eight OpenRouter rows on the board are therefore the extent of what that gateway can contribute to this benchmark; everything else needs the vendor's own key. Reson8 is the first case: the key was added to `.env` the same day, the `reson8` adapter written against the vendor's prerecorded endpoint, and the model run and ranked within the hour.

---

## 5. Results

### 5.1 Leaderboard

Pooled over 12,555 reference words / 47,784 reference characters. S/D/I are word-level counts. Latency is the median wall-clock seconds per 15-minute track on our route.

| # | System | WER | CER | S | D | I | 0fcd | 2bb0 | 86c2 | ea21 | Latency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ElevenLabs Scribe v2 | **1.22 %** | 0.55 % | 67 | 57 | 29 | 1.50 | 1.55 | 0.78 | 1.04 | 14 s |
| 2 | Reson8 | 2.92 % | 1.53 % | 139 | 168 | 60 | 4.18 | 2.91 | 1.78 | 2.88 | 6 s |
| 3 | xAI Grok Voice Transcribe 2.0 | 2.94 % | 1.52 % | 149 | 178 | 42 | 3.92 | 3.24 | 2.22 | 2.37 | 5 s |
| 4 | Google Gemini 3.5 Transcribe | 3.13 % | 1.76 % | 136 | 224 | 33 | 5.13 | 2.97 | 2.09 | 2.37 | 17 s |
| 5 | Meta Muse Voice Transcribe 1.0 | 3.31 % | 1.73 % | 105 | 278 | 33 | 4.21 | 4.00 | 2.40 | 2.61 | 195 s |
| 6 | Deepgram Nova-3 | 3.42 % | 2.05 % | 146 | 175 | 108 | 4.11 | 4.24 | 2.12 | 3.18 | 7 s |
| 7 | Omi General | 3.46 % | 1.88 % | 143 | 190 | 101 | 4.11 | 4.27 | 2.37 | 3.04 | 49 s |
| 8 | Google Chirp 3 | 3.72 % | 2.13 % | 185 | 208 | 74 | 5.13 | 4.09 | 2.12 | 3.58 | 53 s |
| 9 | Microsoft MAI-Transcribe 2 | 3.74 % | 2.58 % | 145 | 187 | 138 | 5.09 | 2.70 | 4.06 | 3.18 | 24 s |
| 10 | NVIDIA Nemotron 3 ASR Streaming (EN) 0.6B · local, 1.12 s | 4.11 % | 2.26 % | 186 | 278 | 52 | 4.83 | 5.09 | 3.19 | 3.28 | 24 s† |
| 11 | NVIDIA Parakeet TDT 0.6B v3 | 4.54 % | 3.15 % | 173 | 175 | 222 | 8.13 | 4.24 | 2.47 | 3.41 | 8 s |
| 12 | OpenAI GPT Transcribe | 4.58 % | 2.79 % | 93 | 456 | 26 | 5.94 | 4.82 | 4.06 | 3.48 | 19 s |
| 13 | Alibaba Qwen3-ASR 1.7B | 5.03 % | 2.37 % | 136 | 209 | 287 | 5.52 | 5.88 | 4.22 | 4.48 | 14 s |
| 14 | OpenAI GPT-4o mini Transcribe | 5.11 % | 3.20 % | 201 | 386 | 55 | 7.02 | 5.30 | 3.65 | 4.51 | 18 s |
| 15 | AssemblyAI Universal-3.5 Pro | 5.13 % | 3.50 % | 114 | 405 | 125 | 7.35 | 3.03 | 6.96 | 3.21 | 23 s |
| 16 | NVIDIA Nemotron 3.5 ASR Streaming 0.6B · local, 1.12 s | 5.87 % | 3.58 % | 213 | 488 | 36 | 7.28 | 6.55 | 4.00 | 5.68 | 31 s† |
| 17 | Mistral Voxtral Mini Transcribe 2 | 6.59 % | 4.74 % | 131 | 529 | 168 | 6.82 | 5.97 | 4.56 | 9.23 | 10 s |
| 18 | OpenAI GPT Realtime Whisper · streaming session | 6.71 % | 4.53 % | 195 | 478 | 170 | 7.15 | 8.40 | 6.46 | 4.68 | 114 s‡ |
| 19 | Cohere Transcribe · local, open weights | 8.51 % | 6.18 % | 123 | 593 | 352 | 7.18 | 7.64 | 6.06 | 13.44 | 15 s† |
| 20 | OpenAI GPT-4o Transcribe | 13.82 % | 10.64 % | 226 | 1,464 | 45 | 25.07 | 13.06 | 10.71 | 6.45 | 18 s |

† Local inference time on an Apple M4 Max (MPS), not an API round trip. ‡ Session wall-clock with audio fed at 10× real time (a 15-minute track takes ≈ 90 s to send), not a batch round trip. Resolution: one WER point is 125.6 words; 0.1 point is about 13 words. Differences under roughly 0.3 points between adjacent systems should be read as ties. Pairs that are effectively tied on this data: Reson8 / Grok Voice Transcribe 2.0 (2 words apart), Gemini 3.5 Transcribe / Muse (23 words), Nova-3 / Omi General (5 words), Chirp 3 / MAI-Transcribe 2 (3 words), Parakeet / GPT Transcribe (5 words), Qwen3-ASR 1.7B / GPT-4o mini Transcribe / Universal-3.5 Pro (10 and 2 words), Voxtral / GPT Realtime Whisper (15 words).

### 5.2 Key findings

1. **Scribe v2 is alone at the top.** 1.22 % is 1.7 points (214 words) ahead of second place, which is itself a tie: Reson8 (2.92 %) and Grok Voice Transcribe 2.0 (2.94 %) are two words apart. Its 153 total errors are spread thinly: the single largest category is 29 fragment deletions, and it transcribes the partner-bleed track with zero cross-talk insertions.
2. **The spread is wider than on short-clip benchmarks.** Twenty models span 1.2–13.8 %, a factor of 11 between best and worst (1.2–8.5 %, a factor of 7, without GPT-4o Transcribe), where published short-clip results for the same vendors typically sit within a factor of two of each other. Long-form conversational audio exposes failure modes short clips never trigger: phrase-level omission (GPT Transcribe, GPT-4o Transcribe, Voxtral), decoder loops (Voxtral without segment decoding, Cohere Transcribe on non-speech pieces), hallucination on short VAD-cut turns (GPT Realtime Whisper), language switching on backchannels (Qwen3-ASR's `嗯`, the GPT-4o family's `ja`), and cross-talk transcription (Parakeet, MAI-Transcribe 2, AssemblyAI).
3. **Deletions dominate for most systems; what gets deleted is conversational.** For 17 of 20 systems D is the largest single category, and for 11 (Gemini, Muse, both Nemotron models, GPT Transcribe, GPT-4o Transcribe, GPT-4o mini, GPT Realtime Whisper, Universal-3.5 Pro, Voxtral, Cohere Transcribe) D exceeds S + I. The exceptions are Scribe v2, where substitutions lead at a very low level, and Parakeet and Qwen3-ASR, where insertions lead. The most-deleted words corpus-wide are `i`, `a`, `and`, `it`, `you`, `know`, `yeah`, `right`, `like`: fragments, repeated function words and backchannels. Several systems appear to apply disfluency cleanup by design: AssemblyAI Universal-3.5 Pro deletes 105 `you`/`know` tokens, roughly 50 of the ≈ 300 "you know" in the corpus (on 86c27418/ch2 that alone is 94 of its 196 errors), MAI-Transcribe 2 drops isolated "yeah"/"right" backchannels (48) and GPT-4o mini Transcribe does the same (42). Nemotron 3.5 ASR behaves the same way (74 deleted `yeah`/`right`), while its English sibling deletes 27. GPT-4o Transcribe's 183 deleted `you`/`know` and 117 deleted `like` are part of its wholesale omission rather than targeted cleanup (Section 5.4f).
4. **The hard track is the quiet one.** 0fcd38f2/ch1 (the male speaker whose channel is gated and whose partner talks 70 % more than he does) is the worst or second-worst track for 10 of 20 systems, and conversation 0fcd38f2 is the worst conversation for 12; insertions of the partner's speech account for 112 of Parakeet's 130 insertions there, and 44–45 for MAI-Transcribe 2 and AssemblyAI (GPT-4o mini inserts one four-word run). Systems that ignore faint background speech (Scribe, Reson8, Grok, Gemini, Chirp 3, Muse, Nova-3, Omi, GPT Transcribe, GPT-4o Transcribe, GPT Realtime Whisper, Qwen, Voxtral, both Nemotron models, Cohere Transcribe) insert nothing from it; Nemotron 3 ASR's 2.78 % on that track is its second-best.
5. **Newer is not always better within a vendor.** OpenAI's rows are the clearest case: GPT Transcribe 4.58 %, GPT-4o mini Transcribe 5.11 %, GPT Realtime Whisper 6.71 % and GPT-4o Transcribe 13.82 %, the last 9.2 points behind the current model on the same audio (Appendix D adds GPT-4o Transcribe Diarize at 5.53 % and open-weight Whisper large-v3 at 6.32 %). Google's two rows are closer: Chirp 3 (3.72 %) is within 0.6 points of Gemini 3.5 Transcribe (3.13 %). The retired siblings in Appendix D are mostly worse than their successors (MAI-Transcribe 1.5 3.97 % against 3.74 %; Qwen3-ASR 0.6B 5.41 % against 5.03 %), with one exception: Grok STT 1.0 (2.82 %) is 13 words better than Grok Voice Transcribe 2.0 (2.94 %), a tie at this resolution.
6. **Within NVIDIA's open models, the English streaming model beats the hosted batch model.** Nemotron 3 ASR (EN) at its 1.12 s setting scores 4.11 % against Parakeet TDT 0.6B v3's 4.54 % with a quarter of the insertions (52 vs 222: no cross-talk, no `mhmm`-style fillers), and the multilingual Nemotron 3.5 ASR trails both at 5.87 %, mostly through deletions (488), which matches NVIDIA's own guidance to prefer the English model for English. Section 5.8 shows how much of each model's accuracy survives at streaming latencies.
7. **Configuration decides whether a score is about the model.** Voxtral scored 107.7 % with a plain `json` request and 6.59 % with `timestamp_granularities=segment`. With OpenAI's default request, GPT-4o Transcribe scored 21.87 % and GPT-4o mini 21.62 % on this data; with `chunking_strategy=auto`, the board's 13.82 % and 5.11 %. Cohere Transcribe scored 11.36 % when every 35 s piece was decoded and 8.51 % with the noise gate its model card recommends; the three pieces the gate removes are silent, and each had been turned into a 65–228-word loop. Section 5.6 documents each.
8. **The newest open-weight model is not the best one.** Cohere Transcribe (2B parameters, released March 2026) ranks 19th at 8.51 %, behind the 0.6B Nemotron and Parakeet models and Qwen3-ASR 1.7B. Its recognition is competitive: 123 substitutions is the second-lowest count on the board, after GPT Transcribe. What costs it is 593 deletions, two-thirds of them fragments, backchannels and discourse markers (Section 5.4b), and 324 of its 352 insertions, which are three fabricated passages on pieces that contain little or no speech but are not quiet enough for a level gate (Section 5.6). Without those three loops it would score 5.93 %, between Nemotron 3.5 ASR and Voxtral.

### 5.3 Per-track word error rate

| System | 0fcd/ch1 | 0fcd/ch2 | 2bb0/ch1 | 2bb0/ch2 | 86c2/ch1 | 86c2/ch2 | ea21/ch1 | ea21/ch2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Scribe v2 | 0.88 | 1.94 | 1.30 | 1.74 | 0.48 | 1.10 | 1.59 | 0.52 |
| Reson8 | 3.34 | 4.76 | 2.67 | 3.10 | 0.79 | 2.84 | 3.46 | 2.33 |
| Grok Voice Transcribe 2.0 | 2.78 | 4.71 | 3.16 | 3.31 | 1.27 | 3.23 | 2.28 | 2.46 |
| Gemini 3.5 Transcribe | 5.17 | 5.09 | 2.61 | 3.26 | 1.27 | 2.97 | 2.14 | 2.59 |
| Muse Voice Transcribe 1.0 | 3.34 | 4.82 | 3.22 | 4.62 | 1.15 | 3.75 | 2.42 | 2.78 |
| Nova-3 | 3.26 | 4.71 | 4.66 | 3.91 | 1.33 | 2.97 | 3.25 | 3.11 |
| Omi General | 2.78 | 5.04 | 4.94 | 3.75 | 1.51 | 3.30 | 3.39 | 2.72 |
| Chirp 3 | 4.38 | 5.65 | 3.84 | 4.29 | 1.33 | 2.97 | 3.25 | 3.89 |
| MAI-Transcribe 2 | 7.32 | 3.54 | 2.26 | 3.04 | 2.78 | 5.43 | 3.11 | 3.24 |
| Nemotron 3 ASR (EN) · 1.12 s | 2.78 | 6.26 | 5.21 | 5.00 | 1.27 | 5.24 | 3.39 | 3.17 |
| Parakeet TDT 0.6B v3 | 12.81 | 4.87 | 3.22 | 5.05 | 1.57 | 3.43 | 3.25 | 3.56 |
| GPT Transcribe | 5.73 | 6.09 | 3.98 | 5.49 | 3.44 | 4.72 | 2.56 | 4.34 |
| Qwen3-ASR 1.7B | 3.18 | 7.14 | 7.27 | 4.78 | 2.96 | 5.56 | 5.32 | 3.69 |
| GPT-4o mini Transcribe | 6.84 | 7.14 | 4.60 | 5.87 | 2.42 | 4.98 | 2.97 | 5.96 |
| Universal-3.5 Pro | 11.61 | 4.37 | 2.74 | 3.26 | 1.63 | 12.67 | 2.63 | 3.76 |
| Nemotron 3.5 ASR · 1.12 s | 7.40 | 7.20 | 6.58 | 6.52 | 2.24 | 5.88 | 4.84 | 6.48 |
| Voxtral Mini Transcribe 2 | 7.64 | 6.26 | 5.01 | 6.74 | 3.56 | 5.62 | 3.18 | 14.90 |
| GPT Realtime Whisper | 6.68 | 7.48 | 11.04 | 6.30 | 3.81 | 9.31 | 3.04 | 6.22 |
| Cohere Transcribe | 7.08 | 7.25 | 8.71 | 6.79 | 5.56 | 6.59 | 22.11 | 5.31 |
| GPT-4o Transcribe | 23.07 | 26.47 | 12.35 | 13.63 | 7.73 | 13.90 | 4.70 | 8.10 |

86c27418/ch1 (a female speaker, 45–55, on the fitness topic) is the easiest track for 14 of 20 systems and below 3.6 % for every system except GPT Realtime Whisper (3.81 %), Cohere Transcribe (5.56 %) and GPT-4o Transcribe (7.73 %). Track difficulty is driven more by the speaker and channel than by the topic: conversation 0fcd38f2 is the hardest for 12 systems although drones and deliveries are the least technical subject in the set. Eleven track scores are above 10 %. Five belong to GPT-4o Transcribe (every track of the first three conversations except 86c27418/ch1), one is GPT Realtime Whisper on 2bb0cfe2/ch1 (11.04 %), and the other five (Parakeet and Universal-3.5 Pro on 0fcd38f2/ch1, Universal-3.5 Pro on 86c27418/ch2, Voxtral on ea217639/ch2, Cohere Transcribe on ea217639/ch1) each have a single identifiable cause: cross-talk, discourse-marker deletion, and repetition loops (Sections 5.4 and 5.6). Cohere Transcribe's 22.11 % on ea217639/ch1 is the clearest case: the track's reference has 1,447 words and the model deletes only 26 and substitutes 19 of them, but two loops add 272 words.

### 5.4 Anatomy of the errors

The alignment in `scores.jsonl` lets every error be attributed. Grouping the 20 runs' substitutions, deletions and insertions by cause:

**(a) Verbatim-reference costs shared by every system.** Fragment restarts ("I-", "s-", "th-") are almost never written by any model; deletions of one- and two-letter fragment tokens range from 29 (Scribe) through 40–140 (most systems) to 162 (GPT Transcribe), 179 (Voxtral) and 190 (Cohere Transcribe), with GPT-4o Transcribe far above at roughly 350 because it drops whole phrases rather than fragments alone. The hyphenated-letter convention ("A-I") costs 10–15 errors per model. These are properties of the benchmark, applied uniformly; they set a floor of roughly 0.3–0.5 % below which no system that writes standard orthography can go.

**(b) Backchannels and discourse markers removed by the system.**

| System | Deleted `yeah` (of 186) | Deleted `right` (of 119) | Deleted `like` (of 383) | Deleted `you`+`know` |
| --- | --- | --- | --- | --- |
| Scribe v2 | 2 | 0 | 0 | 1 |
| Reson8 | 2 | 1 | 0 | 1 |
| Grok Voice Transcribe 2.0 | 3 | — | 2 | 5 |
| Gemini 3.5 Transcribe | 7 | — | 3 | 5 |
| Chirp 3 | 4 | 1 | 5 | 6 |
| MAI-Transcribe 2 | 25 | 23 | 1 | 8 |
| Nemotron 3 ASR (EN) | 16 | 11 | 2 | 7 |
| Nemotron 3.5 ASR | 39 | 35 | 10 | 19 |
| GPT Transcribe | 4 | — | 11 | 11 |
| GPT-4o mini Transcribe | 16 | 26 | 5 | 9 |
| Universal-3.5 Pro | 2 | — | 5 | 105 |
| Voxtral Mini Transcribe 2 | 10 | — | 18 | 22 |
| GPT Realtime Whisper | 3 | 4 | 21 | 60 |
| Cohere Transcribe | 10 | 4 | 30 | 65 |
| GPT-4o Transcribe | 28 | 32 | 117 | 183 |

**(c) Filler spellings the normalizer does not neutralize.** Because the reference's "Mm-hmm" and "uh-huh" vanish in normalization while variant spellings do not, a system's choice of spelling becomes visible as insertions: Deepgram Nova-3 writes `mhmm` (61 insertions, 56 % of its insertions); Omi General writes `uh-huh` → `huh` (54) and Reson8 the same (13, its most frequent insertion) and occasionally `嗯` (6); Qwen3-ASR 1.7B `嗯` (148), `huh` (51), `ha` (20). The symmetric fix (extending the filler list) would move Nova-3 from 3.42 % to about 2.9 % and Qwen3-ASR 1.7B from 5.03 % to about 3.3 %; we report the Whisper-standard behaviour and note the sensitivity rather than tune the normalizer to the systems.

**(d) Language switching.** Qwen3-ASR 1.7B's `嗯` insertions above are backchannels and laughter decoded as Mandarin despite `language=en`; the GPT-4o family shows the same pattern in another language (GPT-4o mini and GPT-4o substitute `ja` for "yeah" 22 and 32 times), and the retired Qwen3-ASR 0.6B produces a 26-cycle `que e que e …` loop on 2bb0cfe2/ch1 (Appendix D). Backchannels and laughter are where multilingual decoders lose the language lock.

**(e) Cross-talk transcription.** Inserted runs of four or more words that match the partner's reference, almost entirely on 0fcd38f2/ch1:

| System | Bleed insertions (words) | Total insertions |
| --- | --- | --- |
| Parakeet TDT 0.6B v3 | 112 | 222 |
| Universal-3.5 Pro | 45 | 125 |
| MAI-Transcribe 2 | 44 | 138 |
| GPT-4o mini Transcribe | 4 | 55 |
| all other systems | 0 | — |

Example, Parakeet on 0fcd38f2/ch1, where the reference (the male speaker) is silent and the inserted text is the female partner's: *"…delivery is absolutely unbelievable they bring my stuff to the v…"*. Whether to count this as an error is a benchmark decision: the words were audible on the channel but were not said by the speaker whose transcript is being scored, so they are counted. It is the clearest measurement in the benchmark of a system's sensitivity to a second, quieter voice.

**(f) Omission.** GPT Transcribe's profile is unusual: the fewest substitutions (93) and insertions (26) of any system, but 456 deletions, in runs of 1–7 words (219 single words, 167 in runs of 2–3, 70 in runs of 4–7): dropped repetitions, dropped "you know", dropped clause-initial "and I". It reads as a fluent transcript that is missing things. Voxtral's 529 deletions are concentrated in fragments and function words; its 168 insertions are dominated by a residual repetition on ea217639/ch2, where one passage is written three times (inserted runs of 51, 19 and 62 words; Section 5.6). GPT-4o Transcribe takes omission to the extreme: 1,464 deletions in runs of up to 20+ words on every track, even with `chunking_strategy=auto` (Section 5.6), which is why it is last on the board by seven points. GPT Realtime Whisper's 170 insertions are a different mechanism, hallucination on short VAD-cut turns, and are analysed in Section 5.7. Cohere Transcribe combines both profiles: 593 deletions of the same conversational kind (190 fragments, 65 `i`, 65 `you`/`know`, 30 `like`, 36 `it`), and 352 insertions of which 324 are three loops on 35 s pieces that hold almost no speech of the scored speaker: a 52-word fabricated sentence ("the only thing that I've ever heard is that I've never heard of the people who are not in the public domain…") produced identically on 2bb0cfe2/ch1 and ea217639/ch1, and a 220-word "and the other side of the road" cycle on ea217639/ch1 (Section 5.6). Outside those three pieces it inserts 28 words in the whole corpus.

**(g) Genuine recognition errors** are the minority for every system. The most common substitutions shared across systems are `in → and`, `was → is`, `the → that`, `a → the`, `could → can`, and `them → em` (MAI-Transcribe 2 and Parakeet write the reduced form). The Nemotron models, Reson8 and Cohere Transcribe add a spelled-acronym variant of the "A-I" cost: they write `AI`, `GPS` and `NPR` as single tokens where the reference has "A-I", "G-P-S" and "N-P-R" (`a → ai` ×6 and `g → gps` ×3 each; Reson8 `n → npr` ×2; Cohere `a → ai` ×5, `g → gps` ×3, `n → npr` ×2, which together with `a → an` ×4 and `haarp → harp` ×2 make up its most frequent substitutions), the same orthographic convention cost paid by every system, in a different direction. Proper nouns are handled well by the leaders; `haarp → harp` and `geez → jeez` are the only recurring content-word confusions.

### 5.5 Aligned examples

Normalized reference against board systems. `[x]` substitution, `<x>` deletion, `{x}` insertion.

**A restart.** Opening of 2bb0cfe2/ch2 (male speaker, news-sources topic); the reference has "and I- I'll listen":

```
REF (Scribe v2, MAI-Transcribe 2, Universal-3.5 Pro identical)
okay okay so the strategy i use is i got these noise canceling headphones right and i put them on
and i i will listen to like the rocky soundtrack and i will just i will walk down the street …

Grok Voice Transcribe 2.0     … and i put [em] on and i <i> <will> listen to like the rocky soundtrack …
Muse Voice Transcribe 1.0     … and i put them on and i <i> <will> listen to like the rocky soundtrack …
Gemini 3.5 / Qwen3-ASR 1.7B   … and i put them on and i i <will> listen to like the rocky soundtrack …
GPT Transcribe / Voxtral      … and i put them on and i <i> will listen to like the rocky soundtrack …
```

Every system recognizes the words; they differ only in whether the restart is written out. Three write the reference verbatim, two collapse "I- I'll" to "I listen", two to "I I listen", two to "I will listen".

**Discourse markers.** 86c27418/ch2, words 250–295 (female speaker, fitness topic):

```
REF (Scribe v2, MAI-Transcribe 2 identical)
that kind of would be my factor there and you know my joints like can they handle a you know
a rigorous r routine that is like hard running and incline running and you know stair stepping
and just different things like that and i

Grok Voice Transcribe 2.0     … a rigorous <r> routine …
GPT Transcribe                … a rigorous <r> routine … like that and <i>
Voxtral / Qwen3-ASR 1.7B      … can they handle <a> you know a rigorous <r> routine …
Universal-3.5 Pro             … and <you> <know> my joints <like> can they handle a <you> <know> <a>
                              rigorous <r> routine … and <you> <know> stair stepping … and <i>
```

Scribe v2 and MAI-Transcribe 2 keep the fragment "r-" ("a rigorous r- routine"); most systems drop it, which is the single most common error type on the board. Universal-3.5 Pro removes every "you know" and the hedging "like", nine words in a 45-word window, which is the behaviour behind its 12.67 % on this track.

### 5.6 Case studies in configuration

**Voxtral Mini Transcribe 2 and the repetition loop.** With the plain `json` request Voxtral transcribes ea217639/ch2 correctly for about three minutes, then emits a 75-word passage ("…and I think that's a great way to incentivize a lot of people…") 179 times, recovers, loops once more, and finishes correctly: 14,262 hypothesis words against a 1,544-word reference, 12,800 insertions, track WER 835 %, corpus WER 107.74 %. The loop is deterministic across retries, with and without `language`, and at `temperature=0`; it does not occur on the original 48 kHz FLAC upload or when the request asks for segment timestamps (`timestamp_granularities=segment`), which makes the server decode segment by segment. The board uses segment decoding: 6.59 %. A residual remains on the same track (1,599 words, 138 insertions, the passage repeated three times, 14.90 %); it is reported as scored. The OpenRouter route for the same model cannot carry the parameter and loops identically (107.72 %), so it is not on the board.

**Cohere Transcribe and silence.** Cohere Transcribe is an attention-based encoder-decoder; its model card notes that, like most such models, it is "eager to transcribe, even non-speech sounds" and recommends a noise gate or voice-activity detector in front of it. The harness feeds each track through the model's own feature extractor, which cuts it into 28–29 pieces of at most 35 s (cut at the quietest point in the last 5 s of each window), and decodes the pieces in batches of 4 with greedy search and `max_new_tokens=256`. Decoding every piece gives 11.36 %. Three pieces, all with a peak level below −38 dBFS (two consecutive pieces at 34–97 s of 0fcd38f2/ch1, where the scored speaker is silent, and one at 609 s of ea217639/ch1), contain no speech of the scored speaker, and each is decoded as a loop that runs to the token limit: "the world is a very important part of the world" (65 words, twice) and "the world is a world of peace" (228 words). The board row applies the recommended gate at its simplest, a peak-level threshold: pieces peaking below −35 dBFS are not sent to the decoder. On this data that is those three pieces plus the 4.5 s trailing stub of 0fcd38f2/ch1, which decoded to nothing anyway (every other piece peaks above −30 dBFS); the gate removes 358 insertions, nothing else changes (S 123 and D 593 are identical), and the score becomes 8.51 %. The ungated run is kept under `runs-extra/cohere-transcribe-ungated`. Three loops survive the gate because the pieces that trigger them are not quiet (peaks of −4 to −15 dBFS): an identical 52-word fabricated sentence on 2bb0cfe2/ch1 (401 s) and ea217639/ch1 (33 s), and a 220-word cycle on ea217639/ch1 (449 s). They account for 324 of the 352 remaining insertions, 2.6 points of the 8.51 %, and are reported as scored; a speech-activity detector, as the model card suggests, rather than a level gate, would be needed to remove them, and the harness does not add one because that would score a pipeline rather than the model. The three pieces that do loop are the same on repeated runs; the ungated and gated runs reproduce to the word.

**OpenAI server-side chunking.** The GPT-4o-family models returned transcripts that stop or skip for long stretches on 15-minute files unless the request sets `chunking_strategy=auto`; the board rows use it:

| Model | Default request | `chunking_strategy=auto` |
| --- | --- | --- |
| GPT-4o Transcribe | 21.87 % (S 113, D 2,563, I 70) | 13.82 % (S 226, D 1,464, I 45) · board |
| GPT-4o mini Transcribe | 21.62 % (S 587, D 1,758, I 370) | 5.11 % (S 201, D 386, I 55) · board |
| GPT-4o Transcribe Diarize | request rejected ("chunking_strategy is required") | 5.53 % (Appendix D) |

Even with the parameter, GPT-4o Transcribe leaves out 1,464 words, 11.7 % of the reference, in runs of up to 20 or more words on every track; this is the model's behaviour on long conversational input, not a request error, and it is reproducible. GPT Transcribe, OpenAI's current model, does not need the parameter and was run without it; its 456 deletions (Section 5.4f) are spread evenly in short runs and are not a truncation effect. The harness supports `chunking_strategy` for any OpenAI-protocol model that does.

**Vendor request limits and client-side chunking.** AssemblyAI via OpenRouter accepts only WAV and returns HTTP 413 above about 4 MB (≈ 125 s of 16 kHz PCM16); the harness sends 8 chunks of ≤ 120 s per track, cut at the quietest point within ± 10 s of each even split. Meta's endpoint accepts ≤ 600 s; 2 chunks of ≤ 540 s. The Nemotron models' Transformers implementation accepts 5,000 encoder frames (400 s) per call; 3 chunks of ≤ 360 s, cut at the quietest point within ± 10 s. Google's Chirp 3 synchronous API accepts 60 s; 17 chunks of ≤ 55 s, cut within ± 5 s. The per-chunk boundaries are stored in each hypothesis's `raw.upload.chunks`. Chunking adds 7, 1, 2 and 16 boundaries per track respectively; inspection of the alignments at boundaries shows no systematic loss, but these systems' scores include whatever boundary effects remain, and a vendor evaluation with a batch endpoint could differ.

**Upload path failures.** Direct uploads of the 42–68 MB FLAC files to xAI, Mistral and OpenAI, and of 29 MB WAV files, repeatedly failed with TLS `bad record mac` errors from the local client; a 14 MB MP3 also failed once (ElevenLabs via OpenRouter, succeeded on resume; MAI-Transcribe 2, one of nine logged failures). All direct `openai-compatible` and `xai` entries therefore use the 64 kbps MP3 upload. The eight remaining MAI-Transcribe 2 failures were a multi-hour upstream outage (HTTP 408 / empty body) on the OpenRouter route; the run completed on resume once the upstream recovered.

### 5.7 Relation to Converse-STT 1.0

Converse-STT 1.0 scored 15 systems on the same four recordings with Cekura's pipeline in September 2026. The leaderboard page carries those 15 on a separate **Version 1 models** tab: each row shows the 1.0 figure and, where the credentials we hold allow it, the same model re-run with the 2.0 harness. The re-runs are ordinary 2.0 runs and are ranked on the 2.0 board as well (Section 4, board policy); the Version 1 tab only adds the side-by-side with the 1.0 figure. Seven of the 15 have been re-run:

| 1.0 system | 1.0 (Cekura) | 2.0 (this harness) | Δ | 2.0 route |
| --- | --- | --- | --- | --- |
| Reson8 | 2.93 % | 2.92 % (2nd) | −0.01 | Reson8 direct, prerecorded endpoint, `filler_mode=verbatim` |
| Deepgram Nova-3 | 4.39 % | 3.42 % (6th) | −0.97 | OpenRouter |
| Google Chirp 3 | 4.19 % | 3.72 % (8th) | −0.47 | OpenRouter, 17 client chunks ≤ 55 s |
| GPT Realtime Whisper | 3.53 % | 6.71 % (18th) | +3.18 | Realtime API transcription session, server VAD, `whisper-1` |
| GPT-4o mini Transcribe | 4.45 % | 5.11 % (14th) | +0.66 | OpenAI direct, `chunking_strategy=auto` |
| AssemblyAI Universal-3.5 Pro | 4.23 % | 5.13 % (15th) | +0.90 | OpenRouter, 8 client chunks ≤ 120 s |
| GPT-4o Transcribe | 12.45 % | 13.82 % (19th) | +1.37 | OpenAI direct, `chunking_strategy=auto` |

Eight could not be re-run. None of them is reachable through OpenRouter (Section 4) and none of their vendors' keys is among the credentials held, so each needs the vendor's own key and, in most cases, a small adapter. Each vendor's public API was checked on 13 October 2026; the table records the API shape the adapter would target and the dependency that blocks the run today. The Version 1 tab states the reason on each un-rerun row.

| 1.0 system | Why it is not on the board | What a run needs | Vendor API shape (adapter effort) |
| --- | --- | --- | --- |
| Deepgram Flux English | Streaming-only; the Deepgram key held is rejected with HTTP 401 on every authenticated endpoint (which is also why Nova-3 is reached through OpenRouter) | A valid Deepgram key | `wss://api.deepgram.com/v2/listen?model=flux-general-en`, 16 kHz linear16 chunks, `EndOfTurn` events; same pattern as the `openai-realtime` adapter (small) |
| Deepgram Flux Multilingual | As above | As above | As above with `model=flux-general-multi` (shares the adapter) |
| Cartesia Ink 2 | No Cartesia key | `CARTESIA_API_KEY` | Batch `POST /stt` multipart plus a WebSocket path; batch is enough for the board (small) |
| Speechmatics Linden | No Speechmatics key | `SPEECHMATICS_API_KEY` | Asynchronous batch jobs API (submit, poll, fetch), like the `omi` adapter (medium) |
| Smallest Pulse | No Smallest key | `SMALLEST_API_KEY` | Batch REST and WebSocket (small–medium) |
| Inworld STT-1 | No Inworld key | `INWORLD_API_KEY` | REST with base64 audio in the request body (medium) |
| Gradium | No Gradium key | `GRADIUM_API_KEY` | WebSocket-only per its documentation (medium) |
| Google Chirp 2 | Not on OpenRouter (`google/chirp-2` and `google/chirp` do not exist there); no Google Cloud credentials | Google Cloud project and service-account credentials | Speech-to-Text v2 batch recognize with `model=chirp_2` (medium) |

Adding a key to `.env` is the only step outside the repository; `models.toml` holds variable names, never values, and the adapter pattern for each API shape already exists in the harness (`openai-compatible` for multipart, `omi` for job APIs, `openai-realtime` for WebSockets).

The numbers are not expected to match: 1.0 used the vendors' own endpoints and Cekura's normalization, while 2.0 reaches Chirp 3, Nova-3 and AssemblyAI through OpenRouter with the chunking described above and uses the Whisper English normalizer. The ordering is stable where the methodology is least different (Reson8 first among the re-run models on both; GPT-4o Transcribe last on both; Nova-3 and Chirp 3 within half a point of each other on both). Reson8 is the closest agreement in the table: 2.93 % with Cekura's pipeline, 2.92 % here, one word apart on 12,555, which is what one expects when the same model is called directly on the same files with no chunking and only the normalizer differs. It was the conversational leader of Converse-STT 1.0 and is second here, two words ahead of Grok Voice Transcribe 2.0 and 214 words behind Scribe v2, a model that was not on the 1.0 board.

The one large disagreement is instructive. GPT Realtime Whisper scores 6.71 % here against 3.53 % in 1.0. Its 2.0 alignments show the characteristic failure of Whisper on short, VAD-cut segments rather than a transcription-quality difference: the Realtime session produced 60–110 turns per track, and on the brief backchannel turns `whisper-1` hallucinates: 48 inserted `ha` tokens (laughter rendered as "ha ha ha…"), 18 inserted `bye` ("Bye." emitted on near-silent turns), and one textbook hallucination, *"I'm going to stop sharing my screen now"*, on 2bb0cfe2/ch1, which is its worst track (11.04 %). Deletions (478) are the usual fragments and discourse markers, but each turn is also decoded without the previous turn's context, which costs clause-initial words. The 1.0 pipeline (Pipecat's VAD, its own turn segmentation and its own normalization) evidently cut the audio differently. Which number is "right" depends on the deployment: a voice agent that uses this API will see our behaviour on backchannel-heavy conversation; a batch deployment of `whisper-1` would not. Both the session settings (`server_vad`, `silence_duration_ms=500`, 10× real-time feed) and every turn boundary are recorded in the run's `hypotheses.jsonl`.

### 5.8 Streaming operating points

The two Nemotron models were re-run at lower lookaheads with nothing else changed (same weights, chunking, prompt and hardware). Each row is a full pass over the eight tracks; the run directories are kept under `runs-extra/` and are scored exactly like board runs, but only the 1.12 s rows are ranked (Section 3.6).

| Model | Lookahead (chunk) | WER | S | D | I | Worst track |
| --- | --- | --- | --- | --- | --- | --- |
| Nemotron 3 ASR (EN) 0.6B | 0 frames (80 ms) | 4.61 % | 235 | 257 | 87 | 0fcd/ch2 6.76 % |
| Nemotron 3 ASR (EN) 0.6B | **13 frames (1.12 s) · board** | **4.11 %** | 186 | 278 | 52 | 0fcd/ch2 6.26 % |
| Nemotron 3.5 ASR 0.6B | 0 frames (80 ms) | 6.99 % | 298 | 542 | 38 | 0fcd/ch1 9.23 % |
| Nemotron 3.5 ASR 0.6B | 3 frames (320 ms) · checkpoint default | 6.18 % | 247 | 488 | 41 | 0fcd/ch2 7.31 % |
| Nemotron 3.5 ASR 0.6B | **13 frames (1.12 s) · board** | **5.87 %** | 213 | 488 | 36 | 0fcd/ch1 7.40 % |

Accuracy improves monotonically with lookahead, as NVIDIA's model cards report for these checkpoints. Going from the 80 ms to the 1.12 s operating point recovers 0.5 points (63 words) for the English model and 1.1 points (141 words) for the multilingual one; the gain is almost entirely in substitutions and insertions (the English model's insertions fall from 87 to 52, its substitutions from 235 to 186), while deletions are flat or rise slightly. In other words, more right context resolves word identity and suppresses false starts, but it does not bring back the backchannels and fragments these models leave out at every latency. Even at 80 ms, the English model (4.61 %) would sit between Parakeet and GPT Transcribe on the board; the 3.5 model at 80 ms (6.99 %) would be 18th, behind GPT Realtime Whisper and ahead of GPT-4o Transcribe. Note that a streaming deployment also pays an endpointing and partial-result cost that this offline measurement does not capture; these rows measure the accuracy of the final transcript at each chunk size, nothing more.

---

## 6. Limitations

- **Size.** Four conversations, eight speakers, 12,555 words. A 0.1-point difference is 13 words; adjacent ranks are often not distinguishable. The board is a reproducible measurement on a fixed public sample, not a population estimate. The private Ocular collection the sample is drawn from is larger; this report covers only the public sample.
- **One channel drives insertions.** Track 0fcd38f2/ch1's gating and bleed account for nearly all cross-talk insertions. A system's insertion count is therefore largely a statement about its behaviour on that one track.
- **Public data.** The recordings have been public since the 1.0 release and cannot be assumed unseen by any system.
- **Route effects.** Eight of the twenty board systems were reached through OpenRouter. The proxy may select among upstream hosts, adds latency, and does not forward every parameter. Five systems were chunked client-side because of request or model limits on their route, and GPT Realtime Whisper is segmented by the server's VAD; a vendor's batch endpoint might behave differently.
- **Local inference.** The two Nemotron rows were produced on our hardware with the 🤗 Transformers port of NVIDIA's checkpoints rather than NVIDIA's NeMo/Riva runtime or hosted NIM. The weights are the same; decoder implementation details (greedy RNNT search, feature extraction) may differ slightly from NVIDIA's reference numbers. The Cohere Transcribe row uses Cohere's own Transformers implementation and chunker in bfloat16 with greedy decoding, plus the level gate described in Section 5.6; a deployment with a voice-activity detector, beam search or a repetition penalty would score differently. The three local rows' "latency" column is compute time, not an API round trip.
- **Board composition.** The board mixes each vendor's current model with older models kept for continuity with Converse-STT 1.0, so a vendor's position depends on which row is read: OpenAI spans 4.58 % (GPT Transcribe) to 13.82 % (GPT-4o Transcribe). Rows are independent runs; no vendor aggregate is computed. Siblings outside both sets are in Appendix D and are not ranked.
- **Configuration sensitivity.** Section 5.6 shows that a single request parameter or preprocessing step can move a score by 3–100 points (on Voxtral, the GPT-4o models and Cohere Transcribe). We used documented defaults plus the minimum settings needed to process long audio, and recorded every parameter; other reasonable choices exist (vocabulary prompts, verbatim modes, temperature) and were not explored.
- **Normalizer.** Whisper's English normalizer removes a fixed filler list and does not neutralize variant spellings (`mhmm`, `huh`, `嗯`), splits hyphenated letter sequences, and maps "oh" to `0`. These are symmetric but visible; Section 5.4(c) quantifies the largest effect (≈ 0.5 points for Nova-3, ≈ 1.7 for Qwen3-ASR 1.7B).
- **Scope.** English only; batch scoring only (streaming models are scored at their batch-quality setting, Section 3.6, with lower-latency points reported but unranked); separated channels only; no diarization, no mixed-channel cpWER, no latency ranking, no cost. These are roadmap items for the harness.
- **Reference quality.** References were human-corrected and audited but are not claimed error-free; a handful of residual reference errors would affect every system equally.

---

## 7. Reproducing a score or adding a system

```bash
uv sync --group dev
cp .env.example .env                      # add the keys you hold; names only ever appear in models.toml
uv run ocular-evals dataset pull          # Hugging Face sample @ pinned revision -> data/converse-en-sample
uv run ocular-evals dataset verify        # SHA-256 + duration of every track
uv run ocular-evals check                 # credentials and dependencies per model
uv run ocular-evals run --model elevenlabs-scribe-v2
uv run ocular-evals report --board converse-stt   # -> reports/converse-stt.html
```

A new OpenAI-protocol model is one `models.toml` entry (`url`, `key` = name of the env var, `model`, `vendor`, optional `upload`, `chunking_strategy`, `timestamp_granularities`, `chunk_s`/`search_s`). Models behind other protocols need an adapter implementing `Provider.transcribe(path, language) -> Hypothesis`; the `xai`, `meta` and `reson8` adapters are 170–200 lines each and are the templates for hosted APIs (`reson8` was written and run on 13 October 2026 in under an hour once the key arrived), `nemotron-asr` and `cohere-transcribe` for local open-weight models (`uv sync --extra nemotron` installs torch and Transformers for both). Changing anything that affects output changes the model digest and requires `--fresh`; changing only transport settings does not. The test suite (`uv run pytest`, 114 tests) runs without network access and covers the manifest builder, normalizer, metrics, chunker, retry logic and every adapter against mocked HTTP responses.

---

## 8. License and citation

The harness is MIT-licensed. The audio and references are Ocular AI's and are released under CC BY 4.0 on Hugging Face; model outputs and scores you produce are yours. The sample is an evaluation set: do not train, fine-tune or distil on it.

```bibtex
@techreport{ocularai2026converse-stt,
  title       = {Converse-STT 2.0: Benchmarking Speech-to-Text on Real Two-Person American English Conversation},
  author      = {{Ocular AI}},
  institution = {Ocular AI},
  year        = {2026},
  month       = {10},
  url         = {https://www.useocular.com/converse/stt},
}
```

Contact: research@useocular.com.

---

## Appendix A. Software versions

Python 3.12.13 · `ocular-evals` 0.1.0 · `jiwer` 4.0.0 (`rapidfuzz` 3.14.6) · `whisper-normalizer` 0.1.12 · `soundfile` 0.14.0 (libsndfile) · `numpy` 2.x · `httpx` 0.28.1 · `huggingface-hub` 1.33.0 · ffmpeg 8.1.1 (MP3 via LAME) · `websockets` 15.0.1 (Realtime API) · `transformers` 5.19.0 and `torch` 2.14.1 (Nemotron and Cohere Transcribe rows, MPS backend on an Apple M4 Max, 36 GB). Dataset manifest built 2026-10-08 06:03 UTC from revision `29b74ad4…`. All runs were made from a single client in the Pacific time zone; the `recorded_at` field of each hypothesis is the authoritative timestamp. Per board row, the UTC window in which the eight tracks were transcribed:

| System | Run directory | Tracks transcribed (UTC) |
| --- | --- | --- |
| Scribe v2 | `elevenlabs-scribe-v2` | 2026-10-11 02:49–03:17 (one upload failure, resumed) |
| Gemini 3.5 Transcribe | `gemini-3-5-transcribe` | 2026-10-11 02:49–02:55 |
| GPT Transcribe | `gpt-transcribe` | 2026-10-11 02:49–02:54 |
| Muse Voice Transcribe 1.0 | `muse-voice-transcribe-1` | 2026-10-11 02:56–03:18 |
| GPT-4o Transcribe | `gpt-4o-transcribe` | 2026-10-11 03:17–03:19 |
| GPT-4o mini Transcribe | `gpt-4o-mini-transcribe` | 2026-10-11 03:17–03:19 |
| Grok Voice Transcribe 2.0 | `grok-voice-transcribe-2` | 2026-10-11 03:21 |
| Nova-3 | `deepgram-nova-3` | 2026-10-11 03:32–03:33 |
| Qwen3-ASR 1.7B | `qwen3-asr-1-7b` | 2026-10-11 03:32–03:34 |
| Omi General | `omi-general-1` | 2026-10-11 03:34–03:40 |
| Parakeet TDT 0.6B v3 | `parakeet-tdt-0-6b-v3` | 2026-10-11 03:35–03:36 |
| Universal-3.5 Pro | `assemblyai-universal-3-5-pro` | 2026-10-11 03:42–03:44 |
| Chirp 3 | `google-chirp-3` | 2026-10-11 03:42–03:49 |
| MAI-Transcribe 2 | `mai-transcribe-2` | 2026-10-11 09:46–09:49 (started 02:48; upstream outage, Section 5.6) |
| Voxtral Mini Transcribe 2 | `voxtral-mini-transcribe-2` | 2026-10-11 10:08–10:09 (segment-decoding configuration) |
| Nemotron 3 ASR Streaming (EN) 0.6B | `nemotron-3-asr-streaming-en-0-6b` | 2026-10-12 21:29–21:36 (local) |
| Nemotron 3.5 ASR Streaming 0.6B | `nemotron-3-5-asr-streaming-0-6b` | 2026-10-12 21:30–21:35 (local) |
| GPT Realtime Whisper | `gpt-realtime-whisper` | 2026-10-13 01:47–02:00 |
| Reson8 | `reson8` | 2026-10-13 02:30–02:31 |
| Cohere Transcribe | `cohere-transcribe` | 2026-10-13 22:32–22:34 (local; gated run) |

The lookahead sweep in Section 5.8 (`runs-extra/`) was run 2026-10-12 21:37–21:44 UTC, immediately after the board rows; the ungated Cohere Transcribe run (`runs-extra/cohere-transcribe-ungated`, Section 5.6) on 2026-10-13 22:30–22:32 UTC; the retired models in Appendix D were run on 2026-10-11. The OpenRouter catalogue probe and the vendor API checks in Sections 4 and 5.7 were made on 2026-10-13.

## Appendix B. Normalization reference cases

| Input | `whisper` profile | `plain` profile |
| --- | --- | --- |
| `Um, uh, oh, mhm, mm-hmm, hmm, yeah, like, you know, okay.` | `0 yeah like you know okay` | `um uh oh mhm mm-hmm hmm yeah like you know okay` |
| `I- I- I think so. Con- consider it.` | `i i i think so con consider it` | same |
| `It's gonna be, like, a hundred percent.` | `it is going to be like a 100%` | `it's gonna be like a hundred percent` |
| `nineteen ninety five` / `twenty twenty-six` | `1995` / `2026` | unchanged |
| `we'll we're won't can't ain't y'all` | `we will we are will not can not aint you all` | unchanged |
| `Dr. Smith … colour vs color` | `doctor smith … color vs color` | `dr smith … colour vs color` |
| `A-I` / `push-ups` / `Mm-hmm` | `a i` / `push ups` / (removed) | `a-i` / `push-ups` / `mm-hmm` |
| `<breath/> <laugh>ha</laugh> [inaudible] (laughs)` | (removed) | (removed) |

## Appendix C. Run-directory schemas

`hypotheses.jsonl` row: `conversation`, `track`, `model`, `model_digest`, `audio_sha256`, `recorded_at`, `latency_s`, `text`, `language`, `duration_s`, `segments[]`, `raw` (full response; for chunked uploads `raw.upload = {format, chunk_s, chunks[{start_s, bytes, request_id}]}`).

`scores.jsonl` track row: `conversation`, `track`, `speaker_id`, `language`, `profile`, `offset_s`, `duration_s`, `latency_s`, `wer{hits, substitutions, deletions, insertions, errors, reference_length, rate}`, `cer{…}`, `reference` (normalized), `hypothesis` (normalized). Conversation row: same counts summed, `tracks`, `complete`.

`summary.json`: `model` (resolved config), `model_digest`, `dataset{root, digest, source}`, `profile`, `primary_metric`, `wer{…}`, `cer{…}`, `tracks{scored, missing, total}`, `failures`, `conversations[]` with per-track scores minus the texts.

## Appendix D. Retired sibling models

These models were run with the same harness, data, normalization and scoring as the board, on 11 October 2026. They are neither a vendor's current model nor a Converse-STT 1.0 model, so under the board policy (Section 4) they are not ranked. Their run directories are archived outside the repository; the `models.toml` entries can be re-created with `ocular-evals models add` to reproduce them.

| Model | Vendor | Board row for the same vendor | WER | CER | S | D | I | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Grok STT 1.0 (`x-ai/grok-stt-1.0`, OpenRouter) | xAI | Grok Voice Transcribe 2.0 | 2.82 % | 1.45 % | 136 | 179 | 39 | Same error profile as 2.0; 13 words better, a tie |
| MAI-Transcribe 1.5 (`microsoft/mai-transcribe-1.5`, OpenRouter) | Microsoft AI | MAI-Transcribe 2 | 3.97 % | 2.72 % | 134 | 259 | 106 | Same backchannel deletions (23 `yeah`, 27 `right`) and 48 bleed insertions |
| Omi Medical (`omi-medical-1`, direct) | Omi | Omi General | 4.02 % | 2.18 % | 158 | 274 | 73 | Domain model; `okay → ok` (23), `huh` insertions (21) |
| Qwen3-ASR 0.6B (`qwen/qwen3-asr-0.6b`, OpenRouter) | Alibaba | Qwen3-ASR 1.7B | 5.41 % | 2.67 % | 156 | 260 | 263 | `嗯` (89), `huh` (27); 26-cycle `que e` loop on 2bb0cfe2/ch1 |
| GPT-4o Transcribe Diarize (`gpt-4o-transcribe-diarize`, direct, `chunking_strategy=auto`) | OpenAI | GPT Transcribe | 5.53 % | 3.46 % | 147 | 490 | 57 | Scored on the plain transcript; deletes `i` 65×, `right` 24×, `yeah` 17× |
| Whisper large-v3 (`openai/whisper-large-v3`, OpenRouter) | OpenAI | GPT Transcribe | 6.32 % | 4.28 % | 139 | 597 | 57 | Open weights; 175 fragment deletions, 60 `you`/`know` deletions |

Per-track WER for the retired models:

| Model | 0fcd/ch1 | 0fcd/ch2 | 2bb0/ch1 | 2bb0/ch2 | 86c2/ch1 | 86c2/ch2 | ea21/ch1 | ea21/ch2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Grok STT 1.0 | 2.86 | 4.71 | 3.16 | 2.99 | 1.03 | 3.04 | 2.07 | 2.46 |
| MAI-Transcribe 1.5 | 8.11 | 4.15 | 2.26 | 2.99 | 2.42 | 5.62 | 3.32 | 3.82 |
| Omi Medical | 4.53 | 4.65 | 4.46 | 4.78 | 3.08 | 3.17 | 3.04 | 4.34 |
| Qwen3-ASR 0.6B | 4.69 | 7.09 | 10.15 | 4.89 | 2.54 | 4.14 | 5.94 | 4.02 |
| GPT-4o Transcribe Diarize | 6.44 | 6.92 | 5.49 | 6.52 | 3.75 | 5.88 | 4.28 | 4.73 |
| Whisper large-v3 | 8.59 | 5.76 | 7.68 | 5.98 | 6.22 | 4.20 | 4.84 | 7.84 |

Two details from the retired table that now belong to board rows: Chirp 3 writes `cuz` for "because" 19 times, and GPT-4o Transcribe's `chunking_strategy` comparison is in Section 5.6.
