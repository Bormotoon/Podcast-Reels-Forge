from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

LOG = logging.getLogger(__name__)

try:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions, RunningMode
    from mediapipe.tasks.python import BaseOptions

    HAS_CV_AND_MP = True
except ImportError:
    HAS_CV_AND_MP = False


_MODEL_PATH = str(
    (Path(__file__).resolve().parents[2] / "assets" / "models" / "blaze_face_short_range.tflite")
)


@dataclass(frozen=True)
class FaceCropSettings:
    samples: int = 7
    min_face_size: int = 60


def face_detection_available() -> bool:
    if not HAS_CV_AND_MP:
        return False
    return Path(_MODEL_PATH).exists()


def _create_detector() -> FaceDetector | None:
    if not HAS_CV_AND_MP or not Path(_MODEL_PATH).exists():
        return None
    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=_MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        min_detection_confidence=0.5,
    )
    return FaceDetector.create_from_options(options)


def detect_face_center_ratio(
    video_path: Path,
    *,
    sample_times_s: list[float],
    settings: FaceCropSettings,
) -> tuple[float | None, float]:

    if not HAS_CV_AND_MP:
        return None, 0.0

    detector = _create_detector()
    if detector is None:
        return None, 0.0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        detector.close()
        return None, 0.0

    ratios: list[float] = []

    try:
        width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        if width <= 0:
            return None, 0.0

        for t in sample_times_s:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            # MediaPipe needs RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = detector.detect(mp_image)

            if result.detections:
                # Get the largest face by bounding box area
                best = max(
                    result.detections,
                    key=lambda d: d.bounding_box.width * d.bounding_box.height,
                )
                bbox = best.bounding_box
                center_x = bbox.origin_x + (bbox.width / 2.0)
                ratios.append(max(0.0, min(1.0, center_x / width)))
    finally:
        cap.release()
        detector.close()

    rate = len(ratios) / len(sample_times_s) if sample_times_s else 0.0
    if not ratios:
        return None, rate

    ratios.sort()
    return ratios[len(ratios) // 2], rate


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
