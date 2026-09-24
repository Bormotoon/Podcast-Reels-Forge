"""RU: Поиск лиц для вертикального кропа.

Что здесь важно для прогона без присмотра:

- модель MediaPipe скачивается при первом использовании (раньше её нужно было
  положить руками, и на свежей установке умный кроп молча выключался);
- мелкие лица на общих планах ищутся ещё и по половинкам кадра: short-range
  модель рассчитана на лица крупным планом;
- лица меньше ``min_face_size`` (постеры, экраны на фоне) не учитываются;
- если в кадре стабильно двое, раскладка ``split`` ставит их друг над другом
  вместо кропа посередине между ними.

EN: Face detection for the vertical crop.

What matters for unattended runs:

- the MediaPipe model is downloaded on first use (it used to have to be put
  in place by hand, and on a fresh install the smart crop silently turned
  itself off);
- small faces in wide shots are also searched for in each half of the frame:
  the short-range model is built for close-ups;
- faces smaller than ``min_face_size`` (posters, screens in the background)
  are ignored;
- when two people are steadily in frame, the ``split`` layout stacks them
  instead of cropping the empty middle between them.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

try:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions, RunningMode

    HAS_CV_AND_MP = True
except ImportError:
    HAS_CV_AND_MP = False


_MODEL_PATH = str(
    (Path(__file__).resolve().parents[2] / "assets" / "models" / "blaze_face_short_range.tflite")
)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)

#: Share of face-bearing samples that must show two separated people for the
#: split layout.
_TWO_FACE_SHARE = 0.6
#: Minimal horizontal distance between the two people, as a share of width.
_MIN_SEPARATION = 0.25


@dataclass(frozen=True)
class FaceCropSettings:
    samples: int = 7
    min_face_size: int = 60
    #: Also search each half of the frame (small faces in wide shots).
    tiled: bool = True


@dataclass(frozen=True)
class Face:
    """A detected face; centre as a share of the frame, size in pixels."""

    cx: float
    cy: float
    width_px: float
    height_px: float


@dataclass
class FaceLayout:
    """What the samples of one clip showed."""

    #: "single", "split" or "none".
    kind: str
    #: Face centres (cx, cy) as shares of the frame: one for single, two
    #: (left first) for split.
    centers: list[tuple[float, float]] = field(default_factory=list)
    #: Share of samples with at least one usable face.
    rate: float = 0.0
    #: The single-crop choice (median of the largest face per sample), also
    #: known for a split layout, for callers that can only crop once.
    primary: tuple[float, float] | None = None


def ensure_face_model(*, url: str = MODEL_URL, path: str = _MODEL_PATH, timeout_s: int = 60) -> bool:
    """Download the detector model if it is missing. False when unavailable."""

    target = Path(path)
    if target.exists() and target.stat().st_size > 0:
        return True
    if os.environ.get("FORGE_NO_MODEL_DOWNLOAD") == "1":
        return False
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        with urllib.request.urlopen(url, timeout=timeout_s) as response, tmp.open("wb") as out:  # noqa: S310
            out.write(response.read())
        if tmp.stat().st_size <= 0:
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(target)
        LOG.info("face detector model downloaded to %s", target)
        return True
    except (OSError, ValueError) as exc:
        LOG.warning("face detector model unavailable (%s): %s", url, exc)
        return False


def face_detection_available(*, download: bool = False) -> bool:
    if not HAS_CV_AND_MP:
        return False
    if Path(_MODEL_PATH).exists():
        return True
    return download and ensure_face_model()


def face_detection_unavailable_reason() -> str:
    if not HAS_CV_AND_MP:
        return "opencv/mediapipe are not installed"
    if not Path(_MODEL_PATH).exists():
        return f"the face model is missing ({_MODEL_PATH}) and could not be downloaded"
    return ""


def _create_detector() -> Any:
    if not HAS_CV_AND_MP or not Path(_MODEL_PATH).exists():
        return None
    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=_MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        min_detection_confidence=0.5,
    )
    return FaceDetector.create_from_options(options)


def _detect(detector: Any, rgb: Any, x_offset: int, frame_w: float, frame_h: float) -> list[Face]:
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    faces: list[Face] = []
    for detection in detector.detect(image).detections or []:
        box = detection.bounding_box
        faces.append(
            Face(
                cx=(x_offset + box.origin_x + box.width / 2.0) / frame_w,
                cy=(box.origin_y + box.height / 2.0) / frame_h,
                width_px=float(box.width),
                height_px=float(box.height),
            ),
        )
    return faces


def merge_faces(faces: list[Face], *, min_distance: float = 0.08) -> list[Face]:
    """Drop duplicates found both in the full frame and in a half."""

    merged: list[Face] = []
    for face in sorted(faces, key=lambda f: -(f.width_px * f.height_px)):
        if all(abs(face.cx - kept.cx) > min_distance for kept in merged):
            merged.append(face)
    return merged


def sample_faces(
    video_path: Path,
    *,
    sample_times_s: list[float],
    settings: FaceCropSettings,
) -> list[list[Face]]:
    """Faces (at least ``min_face_size`` tall) found at each sample time."""

    if not HAS_CV_AND_MP:
        return []
    detector = _create_detector()
    if detector is None:
        return []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        detector.close()
        return []

    samples: list[list[Face]] = []
    try:
        width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            return []
        for t in sample_times_s:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                samples.append([])
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            found = _detect(detector, rgb, 0, width, height)
            if settings.tiled:
                half = int(width) // 2
                for x0 in (0, half):
                    tile = rgb[:, x0 : x0 + half].copy()
                    found += _detect(detector, tile, x0, width, height)
            usable = [f for f in found if f.height_px >= settings.min_face_size]
            samples.append(merge_faces(usable))
    finally:
        cap.release()
        detector.close()
    return samples


def decide_layout(samples: list[list[Face]]) -> FaceLayout:
    """Single median face, two stacked faces, or nothing."""

    total = len(samples)
    with_faces = [faces for faces in samples if faces]
    rate = len(with_faces) / total if total else 0.0
    if not with_faces:
        return FaceLayout(kind="none", rate=rate)

    largest = [max(faces, key=lambda f: f.width_px * f.height_px) for faces in with_faces]
    largest.sort(key=lambda f: f.cx)
    median = largest[len(largest) // 2]
    primary = (median.cx, median.cy)

    pairs: list[tuple[Face, Face]] = []
    for faces in with_faces:
        if len(faces) < 2:
            continue
        top_two = sorted(faces[:2], key=lambda f: f.cx)
        if top_two[1].cx - top_two[0].cx >= _MIN_SEPARATION:
            pairs.append((top_two[0], top_two[1]))
    if len(pairs) >= _TWO_FACE_SHARE * len(with_faces) and len(pairs) >= 2:
        left = sorted(pairs, key=lambda p: p[0].cx)[len(pairs) // 2][0]
        right = sorted(pairs, key=lambda p: p[1].cx)[len(pairs) // 2][1]
        return FaceLayout(
            kind="split",
            centers=[(left.cx, left.cy), (right.cx, right.cy)],
            rate=rate,
            primary=primary,
        )
    return FaceLayout(kind="single", centers=[primary], rate=rate, primary=primary)


def analyze_face_layout(
    video_path: Path,
    *,
    sample_times_s: list[float],
    settings: FaceCropSettings,
) -> FaceLayout:
    return decide_layout(sample_faces(video_path, sample_times_s=sample_times_s, settings=settings))


def detect_face_center_ratio(
    video_path: Path,
    *,
    sample_times_s: list[float],
    settings: FaceCropSettings,
) -> tuple[float | None, float]:
    """Median horizontal face position and the face rate (compatibility API)."""

    layout = analyze_face_layout(video_path, sample_times_s=sample_times_s, settings=settings)
    if layout.primary is None:
        return None, layout.rate
    return layout.primary[0], layout.rate


def build_split_filter(
    *,
    src_w: int,
    src_h: int,
    centers: list[tuple[float, float]],
    target_w: int = 1080,
    target_h: int = 1920,
    zoom: float = 0.9,
) -> str:
    """Filtergraph stacking two speakers: left one on top, right one below.

    Each panel is ``target_w x target_h/2``; the source crop keeps that
    aspect, is ``zoom`` of the frame height, and puts the face a little above
    the panel centre, where a viewer's eye expects it.
    """

    panel_h = target_h // 2
    aspect = target_w / panel_h
    crop_h = int(src_h * max(0.2, min(1.0, zoom)))
    crop_w = int(round(crop_h * aspect))
    if crop_w > src_w:
        crop_w = src_w
        crop_h = int(round(crop_w / aspect))
    crop_w -= crop_w % 2
    crop_h -= crop_h % 2

    parts: list[str] = []
    for label, (cx, cy) in zip(("t", "u"), centers[:2]):
        x = int(round(cx * src_w - crop_w / 2.0))
        y = int(round(cy * src_h - crop_h * 0.45))
        x = max(0, min(src_w - crop_w, x))
        y = max(0, min(src_h - crop_h, y))
        parts.append(f"[{label}0]crop={crop_w}:{crop_h}:{x}:{y},scale={target_w}:{panel_h}[{label}]")
    return (
        "split=2[t0][u0];"
        + ";".join(parts)
        + ";[t][u]vstack=inputs=2"
    )


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
