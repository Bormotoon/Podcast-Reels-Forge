"""RU: Поиск лиц для вертикального кропа.

Что здесь важно для прогона без присмотра:

- детектор — YuNet из OpenCV (``cv2.FaceDetectorYN``): принимает кадр любого
  размера и находит лица от ~20 px, поэтому видит людей и на общем плане.
  Прежний BlazeFace short-range работал на входе 128x128 и на общих планах
  (класс, трое на скамейке) не находил ни одного настоящего лица, зато ловил
  лица на фотографиях со слайдов;
- модель (~230 КБ) скачивается при первом использовании и сверяется по sha256;
- лица меньше ``min_face_size`` (постеры, экраны на фоне) не учитываются;
- если в кадре стабильно двое, раскладка ``split`` ставит их друг над другом
  вместо кропа посередине между ними.

Слежение за лицом во времени и выбор говорящего — в ``face_track.py``.

EN: Face detection for the vertical crop.

What matters for unattended runs:

- the detector is OpenCV's YuNet (``cv2.FaceDetectorYN``): it takes a frame of
  any size and finds faces from ~20 px, so it sees people in wide shots too.
  The former BlazeFace short-range model ran on a 128x128 input and on wide
  shots (a classroom, three people on a bench) found no real face at all while
  it did pick up faces in photos on the slides;
- the model (~230 KB) is downloaded on first use and checked by sha256;
- faces smaller than ``min_face_size`` (posters, screens in the background)
  are ignored;
- when two people are steadily in frame, the ``split`` layout stacks them
  instead of cropping the empty middle between them.

Following faces over time and picking the speaker live in ``face_track.py``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

try:
    import cv2

    HAS_CV = hasattr(cv2, "FaceDetectorYN")
except ImportError:
    HAS_CV = False


_MODEL_PATH = str(
    (Path(__file__).resolve().parents[2] / "assets" / "models" / "face_detection_yunet_2023mar.onnx")
)
MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"

#: Share of face-bearing samples that must show two separated people for the
#: split layout.
_TWO_FACE_SHARE = 0.6
#: Minimal horizontal distance between the two people, as a share of width.
_MIN_SEPARATION = 0.25
#: Detector confidence below which a box is dropped.
MIN_CONFIDENCE = 0.7
#: Frames are downscaled to this width before detection: faces stay above the
#: detector's ~20 px floor in 1080p wide shots, at a quarter of the cost.
DETECT_WIDTH = 960


@dataclass(frozen=True)
class FaceCropSettings:
    samples: int = 7
    min_face_size: int = 60
    #: Legacy (BlazeFace searched each half of the frame); YuNet needs no tiles.
    tiled: bool = True


@dataclass(frozen=True)
class Face:
    """A detected face; centre as a share of the frame, size in pixels."""

    cx: float
    cy: float
    width_px: float
    height_px: float
    score: float = 1.0


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_face_model(
    *, url: str = MODEL_URL, path: str = _MODEL_PATH, sha256: str | None = None, timeout_s: int = 60,
) -> bool:
    """Download the detector model if it is missing. False when unavailable."""

    target = Path(path)
    if target.exists() and target.stat().st_size > 0:
        return True
    if os.environ.get("FORGE_NO_MODEL_DOWNLOAD") == "1":
        return False
    tmp = target.with_name(target.name + ".part")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=timeout_s) as response, tmp.open("wb") as out:  # noqa: S310
            out.write(response.read())
        if tmp.stat().st_size <= 0 or (sha256 and _sha256(tmp) != sha256):
            LOG.warning("face detector model from %s is empty or failed the checksum", url)
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(target)
        LOG.info("face detector model downloaded to %s", target)
        return True
    except (OSError, ValueError) as exc:
        tmp.unlink(missing_ok=True)
        LOG.warning("face detector model unavailable (%s): %s", url, exc)
        return False


def face_detection_available(*, download: bool = False) -> bool:
    if not HAS_CV:
        return False
    if Path(_MODEL_PATH).exists():
        return True
    return download and ensure_face_model(sha256=MODEL_SHA256)


def face_detection_unavailable_reason() -> str:
    if not HAS_CV:
        return "opencv with FaceDetectorYN (>= 4.5.4) is not installed"
    if not Path(_MODEL_PATH).exists():
        return f"the face model is missing ({_MODEL_PATH}) and could not be downloaded"
    return ""


class FaceDetector:
    """YuNet on frames downscaled to ``DETECT_WIDTH``; results in source units."""

    def __init__(self, src_w: int, src_h: int, *, min_confidence: float = MIN_CONFIDENCE) -> None:
        self.src_w = int(src_w)
        self.src_h = int(src_h)
        self.scale = min(1.0, DETECT_WIDTH / float(max(1, self.src_w)))
        self.width = max(2, int(round(self.src_w * self.scale)))
        self.height = max(2, int(round(self.src_h * self.scale)))
        self._net: Any = cv2.FaceDetectorYN.create(
            _MODEL_PATH, "", (self.width, self.height), float(min_confidence), 0.3, 5000,
        )

    def detect(self, bgr: Any) -> list[Face]:
        """Faces in a BGR frame, either at source size or already downscaled."""

        if bgr.shape[1] != self.width or bgr.shape[0] != self.height:
            bgr = cv2.resize(bgr, (self.width, self.height), interpolation=cv2.INTER_AREA)
        _, found = self._net.detect(bgr)
        faces: list[Face] = []
        for row in [] if found is None else found:
            x, y, w, h = (float(v) / self.scale for v in row[:4])
            faces.append(
                Face(
                    cx=(x + w / 2.0) / self.src_w,
                    cy=(y + h / 2.0) / self.src_h,
                    width_px=w,
                    height_px=h,
                    score=float(row[14]),
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


# --------------------------------------------------------------------------
# YuNet on the GPU: the same network and post-processing as
# cv2.FaceDetectorYN, rebuilt in torch from the ONNX weights (read through
# cv2.dnn, so no extra dependency). Frames go straight from the decoder's YUV
# to the GPU; nothing heavy runs on the CPU.
# --------------------------------------------------------------------------

_YUNET_STRIDES = (8, 16, 32)
#: (cls, obj, bbox, kps) head convolutions per stride: 1x1 conv then 3x3.
_YUNET_HEADS = {
    8: ((54, 55), (66, 67), (60, 61), (72, 73)),
    16: ((56, 57), (68, 69), (62, 63), (74, 75)),
    32: ((58, 59), (70, 71), (64, 65), (76, 77)),
}


def _yunet_weights() -> dict[int, tuple[Any, Any]]:
    net = cv2.dnn.readNetFromONNX(_MODEL_PATH)
    weights: dict[int, tuple[Any, Any]] = {}
    for name in net.getLayerNames():
        if not name.startswith("onnx_node!Conv_"):
            continue
        layer_id = net.getLayerId(name)
        weights[int(name.rsplit("_", 1)[1])] = (net.getParam(layer_id, 0), net.getParam(layer_id, 1).reshape(-1))
    return weights


def _build_yunet(device: str) -> Any:
    import torch
    import torch.nn.functional as F
    from torch import nn

    weights = _yunet_weights()

    class YuNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.convs = nn.ModuleDict()
            for index, (w, b) in weights.items():
                out_ch, in_per_group, k, _ = w.shape
                groups = out_ch if (k == 3 and in_per_group == 1 and out_ch > 1) else 1
                conv = nn.Conv2d(
                    in_per_group * groups, out_ch, k, stride=2 if index == 0 else 1,
                    padding=k // 2, groups=groups,
                )
                conv.weight.data = torch.from_numpy(w.copy())
                conv.bias.data = torch.from_numpy(b.copy())  # type: ignore[union-attr]
                self.convs[str(index)] = conv

        def _c(self, index: int, x: Any) -> Any:
            return self.convs[str(index)](x)

        def _unit(self, a: int, x: Any) -> Any:
            # Pointwise conv, depthwise 3x3, ReLU (BatchNorm is folded in).
            return F.relu(self._c(a + 1, self._c(a, x)))

        def forward(self, x: Any) -> list[tuple[Any, Any, Any, Any]]:
            x = F.relu(self._c(0, x))
            x = F.max_pool2d(self._unit(2, x), 2)
            for a in (6, 9, 12, 15):
                x = self._unit(a, x)
            x = F.max_pool2d(x, 2)
            p8 = self._unit(22, self._unit(19, x))
            p16 = self._unit(29, self._unit(26, F.max_pool2d(p8, 2)))
            p32 = self._unit(39, self._unit(36, self._unit(33, F.max_pool2d(p16, 2))))
            f16 = self._unit(45, p16 + F.interpolate(p32, scale_factor=2, mode="nearest"))
            f8 = self._unit(51, p8 + F.interpolate(f16, scale_factor=2, mode="nearest"))
            outs = []
            for stride, feat in ((8, f8), (16, f16), (32, p32)):
                heads = [self._c(b, self._c(a, feat)) for a, b in _YUNET_HEADS[stride]]
                outs.append(tuple(h.flatten(2).transpose(1, 2) for h in heads))
            return outs

    return YuNet().eval().to(device)


def _nms(boxes: Any, scores: Any, threshold: float) -> list[int]:
    """Greedy NMS on a handful of boxes (x, y, w, h)."""

    import numpy as np

    order = np.argsort(-scores)
    keep: list[int] = []
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = x1 + boxes[:, 2], y1 + boxes[:, 3]
    area = boxes[:, 2] * boxes[:, 3]
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        iw = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        ih = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = iw * ih
        iou = inter / np.maximum(area[i] + area[rest] - inter, 1e-6)
        order = rest[iou <= threshold]
    return keep


class GpuFaceDetector:
    """YuNet in torch on ``device`` (CUDA by default).

    ``detect_nv12`` takes an NV12 frame already on the device, as the
    (h * 3 / 2, w) uint8 array ffmpeg writes (Y plane, then interleaved UV —
    the layout NVDEC decodes to, so ffmpeg converts nothing on the CPU).
    """

    _models: dict[str, Any] = {}

    def __init__(
        self, src_w: int, src_h: int, *, device: str = "cuda", min_confidence: float = MIN_CONFIDENCE,
    ) -> None:
        import torch

        self.device = device
        self.src_w = int(src_w)
        self.src_h = int(src_h)
        self.min_confidence = float(min_confidence)
        scale = min(1.0, DETECT_WIDTH / float(max(1, self.src_w)))
        self.width = max(2, int(round(self.src_w * scale)))
        self.height = max(2, int(round(self.src_h * scale)))
        self.pad_w = -(-self.width // 32) * 32
        self.pad_h = -(-self.height // 32) * 32
        if device not in GpuFaceDetector._models:
            GpuFaceDetector._models[device] = _build_yunet(device)
        self.model = GpuFaceDetector._models[device]
        self._torch = torch

    def _bgr_from_nv12(self, nv12: Any) -> Any:
        import torch.nn.functional as F

        torch = self._torch
        h, w = self.src_h - self.src_h % 2, self.src_w - self.src_w % 2
        y = (nv12[:h].float() - 16.0) * 1.164
        chroma = nv12[h:].reshape(h // 2, w // 2, 2).permute(2, 0, 1).float()
        uv = F.interpolate(chroma.unsqueeze(0), size=(h, w), mode="nearest")[0] - 128.0
        b = y + 2.018 * uv[0]
        g = y - 0.391 * uv[0] - 0.813 * uv[1]
        r = y + 1.596 * uv[1]
        bgr = torch.stack([b, g, r]).clamp_(0, 255).unsqueeze(0)
        bgr = F.interpolate(bgr, size=(self.height, self.width), mode="bilinear", antialias=True, align_corners=False)
        return F.pad(bgr, (0, self.pad_w - self.width, 0, self.pad_h - self.height))

    def detect_nv12(self, nv12: Any) -> list[Face]:
        torch = self._torch
        with torch.no_grad():
            heads = self.model(self._bgr_from_nv12(nv12))
            rows = []
            for stride, (cls, obj, bbox, kps) in zip(_YUNET_STRIDES, heads):
                cols = self.pad_w // stride
                score = torch.sqrt(torch.sigmoid(cls[0, :, 0]).clamp(0, 1) * torch.sigmoid(obj[0, :, 0]).clamp(0, 1))
                keep = score >= self.min_confidence
                if not bool(keep.any()):
                    continue
                index = torch.nonzero(keep)[:, 0]
                cx = (index % cols).float()
                cy = (index // cols).float()
                box = bbox[0, index]
                bw = torch.exp(box[:, 2]) * stride
                bh = torch.exp(box[:, 3]) * stride
                bx = (cx + box[:, 0]) * stride - bw / 2
                by = (cy + box[:, 1]) * stride - bh / 2
                rows.append(torch.stack([bx, by, bw, bh, score[index]], dim=1))
            if not rows:
                return []
            found = torch.cat(rows).float().cpu().numpy()
        keep_idx = _nms(found[:, :4], found[:, 4], 0.3)
        scale = self.width / float(self.src_w)
        faces: list[Face] = []
        for i in keep_idx:
            x, y, w, h, score = (float(v) for v in found[i])
            x, y, w, h = x / scale, y / scale, w / scale, h / scale
            faces.append(Face(cx=(x + w / 2) / self.src_w, cy=(y + h / 2) / self.src_h,
                              width_px=w, height_px=h, score=score))
        return faces


def sample_faces(
    video_path: Path,
    *,
    sample_times_s: list[float],
    settings: FaceCropSettings,
) -> list[list[Face]]:
    """Faces (at least ``min_face_size`` tall) found at each sample time."""

    if not face_detection_available():
        return []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    samples: list[list[Face]] = []
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            return []
        detector = FaceDetector(width, height)
        for t in sample_times_s:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                samples.append([])
                continue
            usable = [f for f in detector.detect(frame) if f.height_px >= settings.min_face_size]
            samples.append(merge_faces(usable))
    finally:
        cap.release()
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
