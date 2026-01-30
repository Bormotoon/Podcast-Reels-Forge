from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

LOG = logging.getLogger(__name__)


try:  # Optional dependency
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


@dataclass(frozen=True)
class FaceCropSettings:
    samples: int = 7
    min_face_size: int = 60


def face_detection_available() -> bool:
    return cv2 is not None


def _get_cascades() -> list["cv2.CascadeClassifier"]:  # type: ignore[name-defined]
    if cv2 is None:
        return []

    # Cascade priority: alt2 is usually more robust than default.
    names = [
        "haarcascade_frontalface_alt2.xml",
        "haarcascade_frontalface_default.xml",
    ]
    cascades = []
    try:
        base_path = Path(cv2.data.haarcascades)
    except Exception:
        return []

    for name in names:
        p = base_path / name
        if p.exists():
            c = cv2.CascadeClassifier(str(p))
            if not c.empty():
                cascades.append(c)
    return cascades


def detect_face_center_ratio(
    video_path: Path,
    *,
    sample_times_s: list[float],
    settings: FaceCropSettings,
) -> float | None:
    """Return median face center X as ratio in [0..1], or None.

    Uses Haar cascades (fast, works offline). Picks the largest detected face per frame.
    """

    if cv2 is None:
        return None

    cascades = _get_cascades()
    if not cascades:
        LOG.warning("No opencv Haar cascades found; smart crop disabled")
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    ratios: list[float] = []
    try:
        width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        if width <= 0:
            return None

        for t in sample_times_s:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            found_face = None
            for cascade in cascades:
                faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.05,  # Slightly more precise than 1.1
                    minNeighbors=5,
                    minSize=(settings.min_face_size, settings.min_face_size),
                )
                if faces is not None and len(faces) > 0:
                    # Pick the largest face by area
                    found_face = max(faces, key=lambda r: int(r[2]) * int(r[3]))
                    break

            if found_face is not None:
                x, _, w, _ = found_face
                center_x = float(x) + float(w) / 2.0
                ratios.append(max(0.0, min(1.0, center_x / width)))

    finally:
        cap.release()

    if not ratios:
        LOG.debug("Face detection: 0/%d frames had faces", len(sample_times_s))
        return None

    LOG.debug("Face detection: %d/%d frames had faces", len(ratios), len(sample_times_s))
    ratios.sort()
    return ratios[len(ratios) // 2]


def compute_crop_x_for_scaled_height(
    *,
    src_w: int,
    src_h: int,
    target_w: int,
    target_h: int,
    center_ratio: float,
) -> int:
    """Compute crop X offset after scaling to `target_h` keeping aspect ratio.

    We assume FFmpeg uses `scale=-2:target_h` (width computed automatically).
    """

    if src_w <= 0 or src_h <= 0:
        return 0

    scaled_w = (float(src_w) * float(target_h)) / float(src_h)
    if scaled_w <= target_w:
        return 0

    cx = max(0.0, min(1.0, float(center_ratio))) * scaled_w
    x = int(round(cx - (target_w / 2.0)))
    max_x = int(max(0.0, round(scaled_w - target_w)))
    if x < 0:
        return 0
    if x > max_x:
        return max_x
    return x


def build_sample_times(start_s: float, end_s: float, samples: int) -> list[float]:
    if samples <= 0:
        return []
    duration = max(0.0, float(end_s) - float(start_s))
    if duration <= 0:
        return []
    if samples == 1:
        return [float(start_s) + duration / 2.0]

    # Avoid exact edges (often fades/transitions)
    inner_start = float(start_s) + 0.15 * duration
    inner_end = float(start_s) + 0.85 * duration
    if inner_end <= inner_start:
        inner_start = float(start_s)
        inner_end = float(end_s)

    step = (inner_end - inner_start) / float(samples - 1)
    return [inner_start + step * i for i in range(samples)]
