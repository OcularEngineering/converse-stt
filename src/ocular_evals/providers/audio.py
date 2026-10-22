"""Audio inspection and upload preparation (FLAC passthrough, 16 kHz WAV or FLAC, or MP3 via ffmpeg)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from ocular_evals.providers.base import ProviderError

UPLOAD_FORMATS = ("flac", "wav16k", "flac16k", "mp3")
_CONTENT_TYPES = {
    ".flac": "audio/flac",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".webm": "audio/webm",
}
_TARGET_RATE = 16_000


def content_type_for(path: Path | str) -> str:
    suffix = Path(path).suffix.lower()
    try:
        return _CONTENT_TYPES[suffix]
    except KeyError:
        raise ProviderError(f"unsupported audio type {suffix!r}; expected one of {', '.join(_CONTENT_TYPES)}") from None


def audio_info(path: Path | str) -> tuple[float, int]:
    """Return ``(duration_s, sample_rate)`` without decoding the whole file."""
    try:
        info = sf.info(str(path))
    except Exception as exc:
        raise ProviderError(f"cannot read audio header of {Path(path).name}: {exc}") from exc
    return float(info.duration), int(info.samplerate)


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _workdir(workdir: Path | None) -> Path:
    if workdir is not None:
        workdir.mkdir(parents=True, exist_ok=True)
        return workdir
    return Path(tempfile.mkdtemp(prefix="ocular-evals-"))


def load_mono16k(src: Path) -> np.ndarray:
    """Decode ``src`` to mono float32 samples at 16 kHz, clipped to [-1, 1]."""
    try:
        data, rate = sf.read(str(src), dtype="float32", always_2d=True)
    except Exception as exc:
        raise ProviderError(f"cannot decode {src.name}: {exc}") from exc
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if rate != _TARGET_RATE and mono.size:
        duration = mono.size / float(rate)
        count = max(1, int(round(duration * _TARGET_RATE)))
        src_t = np.arange(mono.size, dtype=np.float64) / float(rate)
        dst_t = np.arange(count, dtype=np.float64) / float(_TARGET_RATE)
        mono = np.interp(dst_t, src_t, mono.astype(np.float64)).astype(np.float32)
    return np.clip(mono, -1.0, 1.0)


def _to_wav16k(src: Path, dst: Path) -> None:
    sf.write(str(dst), load_mono16k(src), _TARGET_RATE, subtype="PCM_16", format="WAV")


def _to_flac16k(src: Path, dst: Path) -> None:
    sf.write(str(dst), load_mono16k(src), _TARGET_RATE, subtype="PCM_16", format="FLAC")


def _quietest_sample(mono: np.ndarray, target: int, *, search: int, frame: int) -> int:
    """Index of the start of the lowest-energy ``frame``-sample window within ``target +- search``."""
    lo = max(0, target - search)
    hi = min(mono.size - frame, target + search)
    if hi <= lo:
        return min(max(target, 0), mono.size)
    squares = np.square(mono[lo : hi + frame].astype(np.float64))
    window = np.convolve(squares, np.ones(frame), mode="valid")  # energy of every frame starting in [lo, hi]
    return lo + int(np.argmin(window)) + frame // 2


def split_wav16k(
    path: Path,
    *,
    max_chunk_s: float,
    search_s: float = 30.0,
    frame_s: float = 0.5,
    workdir: Path | None = None,
) -> list[tuple[Path, float]]:
    """Cut ``path`` into mono 16 kHz PCM16 WAV chunks of at most ``max_chunk_s`` seconds.

    Returns ``[(chunk_path, start_s), ...]`` in order. A file that already fits is written as a
    single chunk. Cut points are spread evenly and then moved to the quietest ``frame_s`` window
    within ``search_s`` seconds of the even split, so a chunk boundary falls in a pause rather
    than in the middle of a word whenever the audio allows it.
    """
    path = Path(path)
    if not path.is_file():
        raise ProviderError(f"audio file not found: {path}")
    if max_chunk_s <= 0:
        raise ProviderError("max_chunk_s must be positive")
    mono = load_mono16k(path)
    target_dir = _workdir(workdir)
    total = mono.size
    max_samples = int(max_chunk_s * _TARGET_RATE)
    count = max(1, -(-total // max_samples)) if total else 1  # ceil division
    search = int(search_s * _TARGET_RATE)
    frame = max(1, int(frame_s * _TARGET_RATE))
    bounds = [0]
    for index in range(1, count):
        target = int(round(total * index / count))
        cut = _quietest_sample(mono, target, search=search, frame=frame)
        # Keep this chunk and everything after it within the limit.
        lowest = max(bounds[-1] + 1, total - (count - index) * max_samples)
        highest = min(total - 1, bounds[-1] + max_samples)
        bounds.append(min(max(cut, lowest), highest))
    bounds.append(total)
    chunks: list[tuple[Path, float]] = []
    for index, (start, end) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        if end - start > max_samples:
            raise ProviderError(
                f"{path.name}: chunk {index} would be {(end - start) / _TARGET_RATE:.1f}s, over the "
                f"{max_chunk_s:.0f}s limit; increase search_s or lower max_chunk_s"
            )
        dst = target_dir / f"{path.stem}.16k.part{index:02d}.wav"
        sf.write(str(dst), mono[start:end], _TARGET_RATE, subtype="PCM_16", format="WAV")
        chunks.append((dst, start / float(_TARGET_RATE)))
    return chunks


def _to_mp3(src: Path, dst: Path) -> None:
    if not ffmpeg_available():
        raise ProviderError(
            "ffmpeg is required to prepare MP3 uploads but was not found on PATH; "
            'install it (e.g. `brew install ffmpeg`) or set upload = "flac" / "wav16k" in the model config'
        )
    command = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", "16000"]
    command += ["-b:a", "64k", str(dst)]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=900, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProviderError(f"ffmpeg failed while encoding {src.name}: {exc}") from exc
    if completed.returncode != 0 or not dst.is_file():
        detail = " ".join(completed.stderr.split())[:300]
        raise ProviderError(f"ffmpeg exited with status {completed.returncode} encoding {src.name}: {detail}")


def prepare_upload(path: Path, *, fmt: str, workdir: Path | None = None) -> tuple[Path, str]:
    """Return ``(upload_path, content_type)`` for ``path`` encoded as ``fmt``.

    ``"flac"`` returns the original file untouched. ``"wav16k"`` writes mono 16 kHz PCM16 WAV,
    ``"flac16k"`` the same samples losslessly compressed as FLAC, and ``"mp3"`` writes mono 16 kHz
    64 kbps MP3 with ffmpeg. Converted files are placed in
    ``workdir`` or a fresh ``ocular-evals-*`` temporary directory.
    """
    path = Path(path)
    if fmt not in UPLOAD_FORMATS:
        raise ProviderError(f"unknown upload format {fmt!r}; expected one of {', '.join(UPLOAD_FORMATS)}")
    if not path.is_file():
        raise ProviderError(f"audio file not found: {path}")
    if fmt == "flac":
        return path, content_type_for(path)
    target_dir = _workdir(workdir)
    if fmt == "wav16k":
        dst = target_dir / f"{path.stem}.16k.wav"
        _to_wav16k(path, dst)
        return dst, "audio/wav"
    if fmt == "flac16k":
        dst = target_dir / f"{path.stem}.16k.flac"
        _to_flac16k(path, dst)
        return dst, "audio/flac"
    dst = target_dir / f"{path.stem}.mp3"
    _to_mp3(path, dst)
    return dst, "audio/mpeg"


__all__ = [
    "UPLOAD_FORMATS",
    "audio_info",
    "content_type_for",
    "ffmpeg_available",
    "load_mono16k",
    "prepare_upload",
    "split_wav16k",
]
