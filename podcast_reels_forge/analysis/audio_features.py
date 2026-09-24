"""RU: Аудио-признаки кандидатов: громкость и паузы.

Текстовые эвристики не слышат эпизод. Смех, эмоциональный всплеск или,
наоборот, вялый кусок с длинными паузами в транскрипте выглядят одинаково.
Здесь ffmpeg меряет отрезок кандидата: среднюю громкость и долю тишины.

Всё опционально: нет исходного аудио, нет ffmpeg, битый вывод — признаки
остаются None, и скоринг подставляет нейтральное значение.

EN: Audio features for candidates: loudness and pauses.

Text heuristics cannot hear the episode. Laughter, an emotional peak, or a
flat stretch padded with long pauses all look the same in a transcript. This
module measures a candidate's span with ffmpeg: mean loudness and how much
of the span is silence.

Everything is optional: with no source audio, no ffmpeg, or unparseable
output the features stay None and scoring falls back to a neutral value.
"""

from __future__ import annotations

import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from podcast_reels_forge.analysis.contracts import MomentRecord, coerce_moment_record
from podcast_reels_forge.utils.ffmpeg import ffmpeg_bin
from podcast_reels_forge.utils.logging_utils import setup_logging

LOGGER = setup_logging()

_MEAN_VOLUME_RE = re.compile(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB")
_MAX_VOLUME_RE = re.compile(r"max_volume:\s*(-?\d+(?:\.\d+)?)\s*dB")
_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")
_SILENCE_DURATION_RE = re.compile(r"silence_duration:\s*(\d+(?:\.\d+)?)")

DEFAULT_NOISE_DB = -30.0
DEFAULT_SILENCE_MIN_S = 0.35
DEFAULT_TIMEOUT_S = 30
DEFAULT_MAX_WORKERS = 4


@dataclass(frozen=True)
class SegmentAudioFeatures:
    """What ffmpeg could tell us about one candidate's span."""

    mean_volume_db: float | None = None
    max_volume_db: float | None = None
    silence_ratio: float | None = None


def parse_volumedetect(stderr: str) -> tuple[float | None, float | None]:
    """Pull mean/max volume out of ffmpeg's volumedetect report."""

    mean_match = _MEAN_VOLUME_RE.search(stderr)
    max_match = _MAX_VOLUME_RE.search(stderr)
    mean_db = float(mean_match.group(1)) if mean_match else None
    max_db = float(max_match.group(1)) if max_match else None
    return mean_db, max_db


def parse_silencedetect(stderr: str, *, span_seconds: float) -> float | None:
    """Fraction of the span ffmpeg reported as silence.

    A silence still running when the clip ends has a ``silence_start`` and no
    duration, so it is measured against the end of the span instead.
    """

    if span_seconds <= 0:
        return None

    durations = [float(value) for value in _SILENCE_DURATION_RE.findall(stderr)]
    starts = [float(value) for value in _SILENCE_START_RE.findall(stderr)]

    total = sum(durations)
    if len(starts) > len(durations) and starts:
        # Unterminated trailing silence: count it up to the end of the span.
        total += max(0.0, span_seconds - starts[-1])

    if not starts and not durations:
        # No silence markers at all is a real measurement: nothing was silent.
        return 0.0 if _MEAN_VOLUME_RE.search(stderr) else None

    return round(min(1.0, total / span_seconds), 4)


def measure_segment_audio(
    source: Path,
    start: float,
    end: float,
    *,
    noise_db: float = DEFAULT_NOISE_DB,
    silence_min_s: float = DEFAULT_SILENCE_MIN_S,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> SegmentAudioFeatures | None:
    """Measure one span with a single audio-only ffmpeg pass."""

    span = end - start
    if span <= 0:
        return None

    command = [
        ffmpeg_bin(),
        "-hide_banner",
        "-nostats",
        "-ss", f"{max(0.0, start):.3f}",
        "-t", f"{span:.3f}",
        "-i", str(source),
        "-vn",
        "-af", f"volumedetect,silencedetect=noise={noise_db}dB:d={silence_min_s}",
        "-f", "null",
        "-",
    ]

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        LOGGER.debug("audio probe failed for [%.1f, %.1f]: %s", start, end, exc)
        return None

    stderr = completed.stderr or ""
    mean_db, max_db = parse_volumedetect(stderr)
    if mean_db is None and max_db is None:
        return None

    return SegmentAudioFeatures(
        mean_volume_db=mean_db,
        max_volume_db=max_db,
        silence_ratio=parse_silencedetect(stderr, span_seconds=span),
    )


def resolve_source_audio(transcript_data: Mapping[str, Any]) -> Path | None:
    """The audio the transcript was produced from, if it is still there."""

    for key in ("source_audio", "audio"):
        raw = transcript_data.get(key)
        if not raw:
            continue
        path = Path(str(raw))
        if path.exists():
            return path
    return None


def _cache_key(
    source: Path,
    start: float,
    end: float,
    *,
    noise_db: float,
    silence_min_s: float,
) -> str:
    try:
        stat = source.stat()
        fingerprint = f"{source.resolve()}:{stat.st_size}:{int(stat.st_mtime)}"
    except OSError:
        fingerprint = str(source)
    return f"{fingerprint}|{start:.2f}|{end:.2f}|{noise_db}|{silence_min_s}"


def _load_cache(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return {str(k): v for k, v in data.items() if isinstance(v, dict)} if isinstance(data, dict) else {}


def _save_cache(path: Path | None, cache: Mapping[str, Mapping[str, Any]]) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        LOGGER.debug("audio feature cache not written: %s", exc)


def annotate_records_with_audio(
    records: Sequence[MomentRecord],
    source: Path | None,
    *,
    enabled: bool = True,
    noise_db: float = DEFAULT_NOISE_DB,
    silence_min_s: float = DEFAULT_SILENCE_MIN_S,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    max_candidates: int = 0,
    max_workers: int = 1,
    cache_path: Path | None = None,
) -> list[MomentRecord]:
    """Attach loudness and silence figures to each candidate.

    Every probe is its own short ffmpeg process, so three things keep the
    cost down:

    * ``max_candidates`` measures only the best-ranked N records (0 = all);
      the rest stay unmeasured and score neutral;
    * probes run on ``max_workers`` threads — ffmpeg start-up dominates a
      short probe, and it parallelizes well;
    * measurements are cached in ``cache_path`` under the source file's
      identity, the span and the detector settings, so a re-run over the same
      episode does not decode the audio again.
    """

    if not enabled or source is None or not records:
        return list(records)

    def _rank(i: int) -> float:
        record = records[i]
        return float(record.priority if record.priority is not None else record.score)

    order = sorted(range(len(records)), key=_rank, reverse=True)
    chosen = set(order[: max_candidates] if max_candidates > 0 else order)

    cache = _load_cache(cache_path)
    keys = {
        i: _cache_key(
            source, records[i].start, records[i].end,
            noise_db=noise_db, silence_min_s=silence_min_s,
        )
        for i in chosen
    }
    missing = [i for i in sorted(chosen) if keys[i] not in cache]

    def _probe(i: int) -> tuple[int, SegmentAudioFeatures | None]:
        return i, measure_segment_audio(
            source,
            records[i].start,
            records[i].end,
            noise_db=noise_db,
            silence_min_s=silence_min_s,
            timeout_s=timeout_s,
        )

    if missing:
        workers = max(1, min(int(max_workers), len(missing)))
        if workers == 1:
            results = [_probe(i) for i in missing]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_probe, missing))
        for i, features in results:
            if features is not None:
                cache[keys[i]] = asdict(features)
        _save_cache(cache_path, cache)

    annotated: list[MomentRecord] = []
    measured = 0
    for i, record in enumerate(records):
        cached = cache.get(keys[i]) if i in keys else None
        if not cached:
            annotated.append(record)
            continue
        payload = {**record.to_dict()}
        if cached.get("mean_volume_db") is not None:
            payload["audio_energy_db"] = cached["mean_volume_db"]
        if cached.get("silence_ratio") is not None:
            payload["audio_silence_ratio"] = cached["silence_ratio"]
        updated = coerce_moment_record(payload)
        annotated.append(updated if updated is not None else record)
        measured += 1

    if measured:
        LOGGER.info(
            "audio features for %d/%d candidates (%d probed with ffmpeg, rest from cache)",
            measured, len(records), len(missing),
        )
    else:
        LOGGER.info("audio features unavailable for all %d candidates", len(records))
    return annotated
