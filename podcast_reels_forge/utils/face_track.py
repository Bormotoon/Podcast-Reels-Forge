"""RU: Вертикальный кадр, который весь клип следит за тем, кто говорит.

Как это устроено (всё офлайн, клип известен целиком):

1. Один проход ffmpeg по клипу: 5 кадров/с уменьшенного видео для детектора
   лиц (YuNet) и, параллельно, ``scdet`` по каждому кадру — склейки планов.
2. Внутри каждого плана лица связываются в треки. Короткие треки (лица на
   слайдах, прохожие) отбрасываются: человек, которого стоит показывать, виден
   заметную часть плана.
3. Если в плане двое и больше, говорящего определяет Light-ASD
   (``active_speaker.py``): он смотрит на губы и слушает звук одновременно.
   Кого показывать, решает Витерби со штрафом за переключение — короткое
   «ага» собеседника не дёргает кадр, а реплика длиннее секунды — переключает.
4. Камера внутри реплики ведёт себя как оператор со штативом: стоит, пока лицо
   в «мёртвой зоне», и плавно (ease-in-out) доводится, когда человек ушёл в
   сторону; цель берётся там, где он остановился, поэтому камера не гонится за
   каждым шагом. Смена говорящего — монтажная склейка (или короткий проезд,
   если люди рядом и выбран ``switch: pan``).
5. Путь кадра становится кусочной функцией от ``t`` в выражении ``crop``:
   рендер остаётся одним проходом ffmpeg, без покадровой перекодировки в Python.

EN: A vertical frame that follows whoever is talking for the whole clip.

How it works (offline, the whole clip is known up front):

1. One ffmpeg pass over the clip: 5 fps of downscaled video for the face
   detector (YuNet) and, alongside, ``scdet`` on every frame for shot cuts.
2. Faces are linked into tracks within each shot. Short tracks (faces on
   slides, passers-by) are dropped: a person worth showing is visible for a
   good part of the shot.
3. With two or more people in a shot, Light-ASD (``active_speaker.py``) tells
   who is talking by watching the lips and listening at once. Who to show is
   decided by Viterbi with a switch penalty, so a short "uh-huh" does not jerk
   the frame while a line longer than a second does.
4. Within a turn the camera behaves like an operator on a tripod: it holds
   while the face stays in a dead zone and glides (ease-in-out) once the person
   has moved; it aims where they stop, so it does not chase every step. A new
   speaker is a cut (or a short pan when people sit close and ``switch: pan``).
5. The camera path becomes a piecewise function of ``t`` in the ``crop``
   expression: rendering stays a single ffmpeg pass.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from podcast_reels_forge.utils import active_speaker
from podcast_reels_forge.utils.face_crop import (
    Face,
    GpuFaceDetector,
    build_split_filter,
    face_detection_available,
    merge_faces,
)
from podcast_reels_forge.utils.ffmpeg import ffmpeg_bin

LOG = logging.getLogger(__name__)

#: Score of a track that is not in frame: never chosen while anyone else is.
_DEAD = -1e9
#: Score of "nobody": below any clipped speaking score, chosen only when no
#: track is alive.
_NOBODY = -4.0
#: Speaking logits are clipped to this range before Viterbi, so one very
#: confident frame cannot outweigh a second of evidence.
_SCORE_CLIP = 3.0


@dataclass(frozen=True)
class TrackingSettings:
    #: Faces smaller than this (source px) are ignored.
    min_face_size: int = 40
    #: Face detection rate.
    detect_fps: float = 5.0
    #: ``scdet`` threshold (0-100); higher finds fewer cuts.
    scene_threshold: float = 10.0
    #: Cuts closer than this to the previous one (or to the clip edges) are
    #: ignored: flashes and fast motion, not new shots.
    min_shot_s: float = 1.0
    #: A face must be found in this share of a shot's samples to be followed.
    min_track_share: float = 0.3
    #: Follow the face within a turn; False holds one framing per turn.
    follow: bool = True
    #: Pick the talking face with Light-ASD when several people are in frame.
    active_speaker: bool = True
    #: "speaker": one face at a time; "split": two steadily visible people
    #: stacked (three or more always use "speaker").
    layout: str = "speaker"
    #: New speaker: "cut" (hard cut) or "pan" (glide when they sit close).
    switch: str = "cut"
    #: Dead zone: how far (share of the output width) the face may drift
    #: from where the camera points before it moves.
    deadzone: float = 0.12
    #: Top glide speed, output widths per second.
    max_speed: float = 0.35
    #: Viterbi penalty for changing the person shown (in logit-samples):
    #: about 0.4 s of a clear lead at 5 fps.
    switch_penalty: float = 10.0
    #: Turns shorter than this are merged into a neighbour.
    min_turn_s: float = 1.0
    #: Switch to the new speaker this much before Viterbi does: the model
    #: reacts a little after the first syllable.
    switch_lead_s: float = 0.2
    #: Torch device for decoding, face detection and the speaker model.
    #: "cuda" never falls back to the CPU: without a GPU the clip keeps the
    #: plain centre crop and a warning says why.
    device: str = "cuda"


@dataclass
class Track:
    """One face followed through a shot (sample indices are clip-wide)."""

    id: int
    shot: int
    idx: list[int] = field(default_factory=list)
    faces: list[Face] = field(default_factory=list)
    #: Light-ASD face crops (112x112 uint8 tensors on the tracking device),
    #: one per 25 fps frame from ``crop_first`` on.
    crops: list[Any] = field(default_factory=list)
    crop_first: int = -1

    @property
    def first(self) -> int:
        return self.idx[0]

    @property
    def last(self) -> int:
        return self.idx[-1]

    def median_cx(self) -> float:
        return float(np.median([f.cx for f in self.faces]))

    def series(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(cx, cy, w, h) at every sample; NaN outside [first, last].

        Gaps are interpolated and every column is median-filtered over five
        samples (one second), which removes detector jitter and the odd box
        that jumps to a neighbour.
        """

        out = np.full((4, n), np.nan)
        span = np.arange(self.first, self.last + 1)
        idx = np.asarray(self.idx, dtype=float)
        cols = (
            [f.cx for f in self.faces],
            [f.cy for f in self.faces],
            [f.width_px for f in self.faces],
            [f.height_px for f in self.faces],
        )
        for row, values in enumerate(cols):
            out[row, span] = _median_filter(np.interp(span, idx, np.asarray(values, dtype=float)), 5)
        return out[0], out[1], out[2], out[3]


@dataclass(frozen=True)
class PathSegment:
    """Camera value over [t0, t1): constant, or an ease-in-out glide v0 -> v1."""

    t0: float
    t1: float
    v0: float
    v1: float


@dataclass
class FramingPlan:
    """How one clip is framed; times are seconds from the clip start."""

    duration: float
    src_w: int
    src_h: int
    #: Left edge (source px) of the full-height window ("speaker" layout).
    x_path: list[PathSegment]
    #: Shots shown as two stacked panels: (t0, t1, [(cx, cy), (cx, cy)]).
    split_shots: list[tuple[float, float, list[tuple[float, float]]]]
    #: Share of samples with a followed face (the face-ratio quality filter).
    rate: float
    cuts: list[float] = field(default_factory=list)
    #: Who was shown when: (t0, t1, track id or -1, speaking score or None).
    turns: list[tuple[float, float, int, float | None]] = field(default_factory=list)
    tracks: list[dict[str, Any]] = field(default_factory=list)
    active_speaker_used: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "duration": round(self.duration, 3),
            "source": [self.src_w, self.src_h],
            "rate": round(self.rate, 3),
            "cuts": [round(c, 3) for c in self.cuts],
            "active_speaker": self.active_speaker_used,
            "tracks": self.tracks,
            "turns": [
                {"start": round(a, 2), "end": round(b, 2), "track": k,
                 "score": None if s is None else round(s, 2)}
                for a, b, k, s in self.turns
            ],
            "split_shots": [
                {"start": round(a, 2), "end": round(b, 2), "centers": [[round(x, 3), round(y, 3)] for x, y in c]}
                for a, b, c in self.split_shots
            ],
            "camera_x": [[round(s.t0, 3), round(s.t1, 3), round(s.v0, 1), round(s.v1, 1)] for s in self.x_path],
        }


# --------------------------------------------------------------------------
# Small numeric helpers
# --------------------------------------------------------------------------


def _median_filter(values: np.ndarray, size: int) -> np.ndarray:
    if values.size < 3 or size < 3:
        return values.copy()
    half = size // 2
    padded = np.pad(values, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, size)
    return np.median(windows, axis=1)


def _moving_average(values: np.ndarray, size: int) -> np.ndarray:
    if values.size == 0 or size <= 1:
        return values.copy()
    half = size // 2
    padded = np.pad(values, half, mode="edge")
    kernel = np.ones(size) / size
    return np.convolve(padded, kernel, mode="valid")[: values.size]


def _ease(v0: float, v1: float, frac: float) -> float:
    frac = min(1.0, max(0.0, frac))
    return v0 + (v1 - v0) * (0.5 - 0.5 * np.cos(np.pi * frac))


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------


def probe_size(video: Path) -> tuple[int, int]:
    try:
        import cv2

        cap = cv2.VideoCapture(str(video))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0))
        cap.release()
        return size
    except Exception:  # noqa: BLE001
        return 0, 0


def _read_exact(stream: Any, size: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def shots_from_cuts(cuts: list[float], duration: float, min_shot_s: float) -> list[tuple[float, float]]:
    kept: list[float] = []
    for cut in sorted(cuts):
        last = kept[-1] if kept else 0.0
        if cut - last >= min_shot_s and duration - cut >= min_shot_s:
            kept.append(cut)
    edges = [0.0, *kept, duration]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


class TrackLinker:
    """Greedy nearest-neighbour linking of faces into tracks, sample by sample."""

    def __init__(self, *, src_w: int, src_h: int, fps: float, max_gap_s: float = 1.0) -> None:
        self.src_w = src_w
        self.src_h = src_h
        self.max_gap = max(1, int(round(max_gap_s * fps)))
        self.tracks: list[Track] = []
        self.active: list[Track] = []

    def update(self, i: int, faces: list[Face], shot: int = 0) -> None:
        self.active = [t for t in self.active if t.shot == shot and i - t.last <= self.max_gap]
        pairs: list[tuple[float, int, int]] = []
        for fi, face in enumerate(faces):
            for ti, track in enumerate(self.active):
                ref = track.faces[-1]
                dist = float(np.hypot((face.cx - ref.cx) * self.src_w, (face.cy - ref.cy) * self.src_h))
                scale = max(face.height_px, ref.height_px, 1.0)
                # Up to ~1.5 face heights per step, more after a gap.
                gate = 1.5 * scale * (1 + 0.5 * (i - track.last - 1))
                if dist <= gate and 0.5 <= face.height_px / max(ref.height_px, 1.0) <= 2.0:
                    pairs.append((dist / scale, fi, ti))
        used_faces: set[int] = set()
        used_tracks: set[int] = set()
        for _cost, fi, ti in sorted(pairs):
            if fi in used_faces or ti in used_tracks:
                continue
            used_faces.add(fi)
            used_tracks.add(ti)
            self.active[ti].idx.append(i)
            self.active[ti].faces.append(faces[fi])
        for fi, face in enumerate(faces):
            if fi not in used_faces:
                track = Track(id=len(self.tracks), shot=shot, idx=[i], faces=[face])
                self.tracks.append(track)
                self.active.append(track)


def is_still_picture(
    track: Track, *, src_w: int, src_h: int, median_step: float = 0.009, p90_step: float = 0.035,
) -> bool:
    """A face that never moves: a photo on a slide, a poster, a portrait.

    A living head moves a little even when its owner sits still: on POS
    footage people moved 0.013-0.08 of their face height per 0.2 s (median),
    faces on slides 0.003-0.008. Both the median and the 90th percentile must
    be small, so one still stretch does not hide a person.
    """

    if len(track.idx) < 10:
        return False
    x = np.array([f.cx for f in track.faces]) * src_w
    y = np.array([f.cy for f in track.faces]) * src_h
    height = max(float(np.median([f.height_px for f in track.faces])), 1.0)
    steps = np.hypot(np.diff(x), np.diff(y)) / height
    return float(np.median(steps)) < median_step and float(np.percentile(steps, 90)) < p90_step


def build_tracks(
    samples: list[list[Face]],
    sample_shot: list[int],
    *,
    src_w: int,
    src_h: int,
    fps: float,
    max_gap_s: float = 1.0,
) -> list[Track]:
    """Link faces into tracks; a track never crosses a shot cut."""

    linker = TrackLinker(src_w=src_w, src_h=src_h, fps=fps, max_gap_s=max_gap_s)
    for i, faces in enumerate(samples):
        linker.update(i, faces, sample_shot[i])
    return linker.tracks


def split_at_shots(tracks: list[Track], sample_shot: list[int], frames_per_sample: int) -> list[Track]:
    """Cut tracks linked across a shot change (cuts are known only after the scan)."""

    out: list[Track] = []
    for track in tracks:
        groups: dict[int, list[int]] = {}
        for pos, i in enumerate(track.idx):
            groups.setdefault(sample_shot[i], []).append(pos)
        for shot, positions in groups.items():
            idx = [track.idx[p] for p in positions]
            piece = Track(id=len(out), shot=shot, idx=idx, faces=[track.faces[p] for p in positions])
            if track.crops:
                f0 = max(track.crop_first, idx[0] * frames_per_sample)
                f1 = min(track.crop_first + len(track.crops), (idx[-1] + 1) * frames_per_sample)
                if f1 > f0:
                    piece.crop_first = f0
                    piece.crops = track.crops[f0 - track.crop_first : f1 - track.crop_first]
            out.append(piece)
    return out


def _readinto_exact(stream: Any, view: memoryview) -> bool:
    filled = 0
    total = len(view)
    while filled < total:
        got = stream.readinto(view[filled:])
        if not got:
            return False
        filled += got
    return True


def crop_grid(boxes: Any, frame_w: int, frame_h: int) -> Any:
    """grid_sample grid for Light-ASD crops of several faces at once.

    ``boxes`` is a (K, 4) tensor of (x0, y0, x1, y1) source rectangles from
    :func:`active_speaker.face_crop_box`. The reference pipeline resizes that
    rectangle to 224x224 (linear) and keeps the centre 112x112; sampling those
    112 pixel centres directly gives the same picture in one pass. Returns a
    (1, K * 112, 112, 2) grid: one batch, the faces stacked along the height.
    """

    import torch

    k = boxes.shape[0]
    j = torch.arange(56, 168, device=boxes.device, dtype=torch.float32) + 0.5
    x0, y0, x1, y1 = boxes[:, 0:1], boxes[:, 1:2], boxes[:, 2:3], boxes[:, 3:4]
    px = x0 + j[None, :] * (x1 - x0) / 224.0 - 0.5
    py = y0 + j[None, :] * (y1 - y0) / 224.0 - 0.5
    gx = (2 * px + 1) / frame_w - 1
    gy = (2 * py + 1) / frame_h - 1
    grid = torch.stack(torch.broadcast_tensors(gx[:, None, :], gy[:, :, None]), dim=-1)
    return grid.reshape(1, k * 112, 112, 2)


def scan_clip(
    video: Path,
    start: float,
    end: float,
    detector: GpuFaceDetector,
    settings: TrackingSettings,
    *,
    collect_crops: bool,
) -> tuple[list[list[Face]], list[float], list[Track]]:
    """One GPU decode of the clip: faces, shot cuts and face crops.

    ffmpeg decodes on NVDEC and hands over every frame at 25 fps as NV12 —
    the decoder's own format, so the CPU only copies bytes. Each frame goes
    to the GPU once, and everything else happens there: every
    ``25 / detect_fps``-th frame is searched for faces, every frame gives
    each live track a crop for the speaker model (one ``grid_sample`` for all
    faces; the crops stay on the GPU) and feeds the shot-cut detector — the
    ``scdet`` measure (mean frame difference and its jump) on a 320x180 luma
    thumbnail. Returns (faces per sample, cuts in clip seconds, tracks — not
    yet split at cuts).
    """

    import torch
    import torch.nn.functional as F

    device = settings.device
    vfps = active_speaker.VIDEO_FPS
    step = max(1, int(round(vfps / settings.detect_fps)))
    w, h = detector.src_w - detector.src_w % 2, detector.src_h - detector.src_h % 2
    even = "" if (w, h) == (detector.src_w, detector.src_h) else f",crop={w}:{h}:0:0"
    cmd = [
        ffmpeg_bin(), "-hide_banner", "-nostdin", "-nostats", "-v", "error",
        *(["-hwaccel", "cuda"] if device.startswith("cuda") else []),
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(video), "-an",
        "-vf", f"fps={vfps}{even}", "-pix_fmt", "nv12", "-f", "rawvideo", "pipe:1",
    ]
    linker = TrackLinker(src_w=w, src_h=h, fps=vfps / step)
    samples: list[list[Face]] = []
    frame_bytes = w * h * 3 // 2
    # Frames travel in batches of one detection step: one host-to-GPU copy,
    # one thumbnail pass, one detection and one grid_sample per batch.
    host = torch.empty(frame_bytes * step, dtype=torch.uint8)
    if device.startswith("cuda"):
        host = host.pin_memory()
    view = memoryview(host.numpy())  # type: ignore[arg-type]
    mafd_parts: list[Any] = []
    prev_thumb = None
    with tempfile.TemporaryFile() as err:
        # stderr goes to a file: a full pipe would stall ffmpeg while we read stdout.
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
        assert proc.stdout is not None
        frame = 0
        try:
            with torch.no_grad():
                while True:
                    got = 0
                    while got < step and _readinto_exact(
                        proc.stdout, view[got * frame_bytes : (got + 1) * frame_bytes],
                    ):
                        got += 1
                    if got == 0:
                        break
                    batch = host[: got * frame_bytes].to(device).view(got, h * 3 // 2, w)
                    luma = batch[:, :h]
                    thumbs = F.interpolate(luma[:, None].float(), size=(180, 320), mode="area")
                    chain = thumbs if prev_thumb is None else torch.cat([prev_thumb, thumbs])
                    if chain.shape[0] > 1:
                        mafd_parts.append((chain[1:] - chain[:-1]).abs().mean(dim=(1, 2, 3)) * (100.0 / 256.0))
                    prev_thumb = thumbs[-1:]

                    found = [f for f in detector.detect_nv12(batch[0]) if f.height_px >= settings.min_face_size]
                    faces = merge_faces(found, min_distance=0.02)
                    samples.append(faces)
                    linker.update(frame // step, faces)

                    if collect_crops and linker.active:
                        live = list(linker.active)
                        boxes = torch.tensor(
                            [
                                active_speaker.face_crop_box(
                                    t.faces[-1].cx * w, t.faces[-1].cy * h,
                                    max(t.faces[-1].width_px, t.faces[-1].height_px) / 2.0,
                                )
                                for t in live
                            ],
                            device=device, dtype=torch.float32,
                        )
                        grid = crop_grid(boxes, w, h).expand(got, -1, -1, -1)
                        gray = luma[:, None].float().sub_(110.0)
                        sampled = F.grid_sample(gray, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
                        crops = sampled.add_(110.0).round_().clamp_(0, 255).to(torch.uint8)
                        crops = crops.view(got, len(live), 112, 112)
                        for k, t in enumerate(live):
                            if t.crop_first < 0:
                                t.crop_first = frame
                            t.crops.extend(crops[:, k].unbind(0))
                    frame += got
                    if got < step:
                        break
        finally:
            proc.stdout.close()
            proc.wait()
        if frame == 0:
            err.seek(0)
            LOG.warning("face tracking: ffmpeg gave no frames for %s: %s",
                        video.name, err.read().decode("utf-8", "replace").strip()[-400:])
    mafd = torch.cat(mafd_parts).cpu().numpy() if mafd_parts else np.zeros(0)
    return samples, scene_cuts(mafd, settings.scene_threshold, fps=vfps), linker.tracks


def scene_cuts(mafd: np.ndarray, threshold: float, *, fps: float) -> list[float]:
    """Cut times from per-frame mean absolute frame differences (0-100).

    The ffmpeg ``scdet`` score: min(mafd, |mafd - previous mafd|) — a cut is
    a sudden difference, not steady motion. ``mafd[k]`` compares frames k and
    k + 1, so a cut found there starts at frame k + 1.
    """

    cuts: list[float] = []
    prev = 0.0
    for k, value in enumerate(mafd):
        score = min(float(value), abs(float(value) - prev))
        prev = float(value)
        if score >= threshold:
            cuts.append((k + 1) / fps)
    return cuts


def speaking_by_sample(
    video: Path,
    start: float,
    end: float,
    tracks: list[Track],
    *,
    n_samples: int,
    frames_per_sample: int,
    device: str,
) -> dict[int, np.ndarray]:
    """Per-sample speaking logits of tracks that carry crops (NaN elsewhere)."""

    import torch

    usable = [t for t in tracks if len(t.crops) >= active_speaker.VIDEO_FPS // 2]
    if not usable:
        return {}
    audio = subprocess.run(
        [
            ffmpeg_bin(), "-hide_banner", "-nostdin", "-v", "error",
            "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(video), "-vn",
            "-ac", "1", "-ar", str(active_speaker.SAMPLE_RATE), "-f", "s16le", "pipe:1",
        ],
        capture_output=True,
        check=False,
    ).stdout
    if not audio:
        return {}
    feats = active_speaker.mfcc(np.frombuffer(audio, np.int16))

    result: dict[int, np.ndarray] = {}
    for track in usable:
        frames = active_speaker.speaking_scores(
            feats, torch.stack(track.crops), track.crop_first, device=device,
        )
        per_sample = np.full(n_samples, np.nan)
        sample_of_frame = (track.crop_first + np.arange(frames.size)) // frames_per_sample
        for s in np.unique(sample_of_frame):
            if 0 <= s < n_samples:
                per_sample[s] = float(frames[sample_of_frame == s].mean())
        result[track.id] = per_sample
    return result


def choose_speakers(
    emissions: np.ndarray,
    *,
    switch_penalty: float,
) -> np.ndarray:
    """Viterbi over (states, samples): best state per sample.

    The last state is "nobody". Switching between any two states costs
    ``switch_penalty``.
    """

    n_states, n = emissions.shape
    if n == 0:
        return np.zeros(0, dtype=int)
    score = emissions[:, 0].copy()
    back = np.zeros((n_states, n), dtype=int)
    for i in range(1, n):
        best_prev = int(np.argmax(score))
        switch_score = score[best_prev] - switch_penalty
        stay = score >= switch_score
        back[:, i] = np.where(stay, np.arange(n_states), best_prev)
        score = np.where(stay, score, switch_score) + emissions[:, i]
    path = np.zeros(n, dtype=int)
    path[-1] = int(np.argmax(score))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[path[i], i]
    return path


def _runs(path: np.ndarray) -> list[tuple[int, int, int]]:
    """(state, first, end) runs of equal values."""

    runs: list[tuple[int, int, int]] = []
    start = 0
    for i in range(1, path.size + 1):
        if i == path.size or path[i] != path[start]:
            runs.append((int(path[start]), start, i))
            start = i
    return runs


def merge_short_turns(path: np.ndarray, alive: np.ndarray, min_len: int) -> np.ndarray:
    """Give turns shorter than ``min_len`` samples to a neighbour that is in frame."""

    path = path.copy()
    changed = True
    while changed:
        changed = False
        runs = _runs(path)
        if len(runs) < 2:
            break
        for k, (state, a, b) in enumerate(runs):
            if b - a >= min_len:
                continue
            for neighbour in (runs[k - 1] if k > 0 else None, runs[k + 1] if k + 1 < len(runs) else None):
                if neighbour is None or neighbour[0] == state:
                    continue
                if bool(np.all(alive[neighbour[0], a:b])):
                    path[a:b] = neighbour[0]
                    changed = True
                    break
            if changed:
                break
    return path


# --------------------------------------------------------------------------
# Camera path
# --------------------------------------------------------------------------


def plan_follow(
    times: np.ndarray,
    xs: np.ndarray,
    *,
    start_time: float,
    start_value: float,
    end_time: float,
    deadzone: float,
    max_speed: float,
    follow: bool,
    settle_s: float = 0.6,
    lead_s: float = 0.2,
    min_glide_s: float = 0.8,
    quiet_tail_s: float = 1.0,
) -> list[PathSegment]:
    """Tripod-style camera path for one turn.

    Holds while the target stays within ``deadzone`` of the camera; once it
    leaves, glides (ease-in-out) to where the target settles — the first
    stretch of ``settle_s`` that moves less than half the dead zone — starting
    ``lead_s`` early, since the whole clip is known. No glide starts in the
    last ``quiet_tail_s`` of the turn.
    """

    t_start = float(start_time)
    c = float(start_value)
    if not follow or xs.size < 2:
        return [PathSegment(t_start, end_time, c, c)]
    step = float(times[1] - times[0]) if times.size > 1 else 0.2
    settle_n = max(1, int(round(settle_s / step)))
    segments: list[PathSegment] = []
    hold_from = t_start
    i = 0
    n = xs.size
    while i < n:
        if abs(xs[i] - c) <= deadzone:
            i += 1
            continue
        j = i
        while j < n - 1:
            window = xs[j : j + settle_n + 1]
            if float(window.max() - window.min()) < deadzone * 0.5:
                break
            j += 1
        dest = float(np.median(xs[j : j + settle_n + 1]))
        if abs(dest - c) <= deadzone * 0.5:
            i = j + 1
            continue
        glide_start = max(hold_from, float(times[i]) - lead_s)
        if glide_start >= end_time - quiet_tail_s:
            # A move just before the next cut only distracts.
            break
        duration = max(min_glide_s, abs(dest - c) / max(max_speed, 1e-6))
        glide_end = min(end_time, max(glide_start + duration, float(times[j])))
        if glide_end - glide_start < 0.1:
            break
        if glide_start > hold_from:
            segments.append(PathSegment(hold_from, glide_start, c, c))
        segments.append(PathSegment(glide_start, glide_end, c, dest))
        c = dest
        hold_from = glide_end
        i = int(np.searchsorted(times, glide_end))
    if end_time > hold_from:
        segments.append(PathSegment(hold_from, end_time, c, c))
    return _merge_chained_glides(segments)


def _merge_chained_glides(segments: list[PathSegment]) -> list[PathSegment]:
    """Back-to-back glides in one direction become one smooth glide."""

    out: list[PathSegment] = []
    for seg in segments:
        if out:
            prev = out[-1]
            prev_dir = np.sign(prev.v1 - prev.v0)
            seg_dir = np.sign(seg.v1 - seg.v0)
            if prev_dir != 0 and prev_dir == seg_dir and abs(seg.t0 - prev.t1) < 1e-6:
                out[-1] = PathSegment(prev.t0, seg.t1, prev.v0, seg.v1)
                continue
            if prev_dir == 0 and seg_dir == 0 and abs(prev.v0 - seg.v0) < 0.5:
                out[-1] = PathSegment(prev.t0, seg.t1, prev.v0, prev.v0)
                continue
        out.append(seg)
    return out


def path_value(segments: list[PathSegment], t: float) -> float:
    for seg in segments:
        if seg.t0 <= t < seg.t1:
            span = seg.t1 - seg.t0
            return _ease(seg.v0, seg.v1, (t - seg.t0) / span if span > 0 else 1.0)
    return segments[-1].v1 if segments else 0.0


def piecewise_expr(segments: list[PathSegment], decimals: int = 1) -> str:
    """ffmpeg expression of ``t`` for a camera path (first/last open-ended)."""

    if not segments:
        return "0"
    terms: list[str] = []
    last = len(segments) - 1
    for k, seg in enumerate(segments):
        v0 = round(seg.v0, decimals)
        dv = round(seg.v1 - seg.v0, decimals)
        if dv == 0:
            value = f"{v0:g}"
        else:
            span = max(seg.t1 - seg.t0, 1e-3)
            value = f"({v0:g}+({dv:g})*(0.5-0.5*cos(PI*(t-{seg.t0:.3f})/{span:.3f})))"
        conds = []
        if k > 0:
            conds.append(f"gte(t,{seg.t0:.3f})")
        if k < last:
            conds.append(f"lt(t,{seg.t1:.3f})")
        terms.append("*".join([*conds, value]))
    return "+".join(terms)


# --------------------------------------------------------------------------
# The whole clip
# --------------------------------------------------------------------------


def crop_width(src_w: int, src_h: int, out_w: int = 1080, out_h: int = 1920) -> int:
    """Width of the full-height source window with the output's aspect (even)."""

    width = int(round(out_w * src_h / out_h))
    return width - width % 2


def _crop_x(cx: np.ndarray | float, src_w: int, crop_w: int) -> np.ndarray:
    """Left edge (source px) of a window of ``crop_w`` centred on ``cx``."""

    max_x = max(0.0, float(src_w - crop_w))
    return np.clip(np.asarray(cx, dtype=float) * src_w - crop_w / 2.0, 0.0, max_x)


def _cuda_ready() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def analyze_clip(
    video: Path,
    start: float,
    end: float,
    settings: TrackingSettings,
    *,
    out_w: int = 1080,
    out_h: int = 1920,
) -> FramingPlan | None:
    """Plan the vertical framing of ``video`` between ``start`` and ``end``.

    None when faces cannot be detected at all (no model, unreadable video).
    """

    if not face_detection_available():
        return None
    src_w, src_h = probe_size(video)
    if src_w <= 0 or src_h <= 0:
        return None
    duration = max(0.0, float(end) - float(start))
    step = max(1, int(round(active_speaker.VIDEO_FPS / settings.detect_fps)))
    fps = active_speaker.VIDEO_FPS / step
    want_speaker = settings.active_speaker and active_speaker.asd_available(download=True)
    if settings.device.startswith("cuda") and not _cuda_ready():
        LOG.warning("face tracking needs a CUDA GPU and none is usable; centre crop for this clip")
        return None
    detector = GpuFaceDetector(src_w, src_h, device=settings.device)
    samples, raw_cuts, linked = scan_clip(video, start, end, detector, settings, collect_crops=want_speaker)
    n = len(samples)
    if n == 0:
        return None
    times = np.arange(n) / fps
    shots = shots_from_cuts(raw_cuts, duration, settings.min_shot_s)
    shot_ends = [b for _a, b in shots]
    sample_shot = [min(int(np.searchsorted(shot_ends, t, side="right")), len(shots) - 1) for t in times]
    tracks = split_at_shots(linked, sample_shot, step)
    shot_samples = [sum(1 for s in sample_shot if s == k) for k in range(len(shots))]
    strong = [
        t for t in tracks
        if len(t.idx) >= max(3, settings.min_track_share * shot_samples[t.shot])
        and not is_still_picture(t, src_w=src_w, src_h=src_h)
    ]

    crop_w = crop_width(src_w, src_h, out_w, out_h)
    center_x = float(_crop_x(0.5, src_w, crop_w))

    # Shots with two well separated, steadily visible people may be stacked.
    split_shots: list[tuple[float, float, list[tuple[float, float]]]] = []
    split_ids: set[int] = set()
    if settings.layout == "split":
        for k, (a, b) in enumerate(shots):
            mine = sorted((t for t in strong if t.shot == k), key=lambda t: -len(t.idx))
            if len(mine) != 2 or min(len(t.idx) for t in mine) < 0.6 * shot_samples[k]:
                continue
            left, right = sorted(mine, key=lambda t: t.median_cx())
            if right.median_cx() - left.median_cx() < 0.25:
                continue
            centers = [
                (left.median_cx(), float(np.median([f.cy for f in left.faces]))),
                (right.median_cx(), float(np.median([f.cy for f in right.faces]))),
            ]
            split_shots.append((a, b, centers))
            split_ids.add(k)

    # Who is talking, where several people share a shot.
    speaking: dict[int, np.ndarray] = {}
    crowded = [
        t for t in strong
        if t.shot not in split_ids and sum(1 for o in strong if o.shot == t.shot) >= 2
    ]
    if crowded and want_speaker:
        try:
            speaking = speaking_by_sample(
                video, start, end, crowded, n_samples=n, frames_per_sample=step, device=settings.device,
            )
        except Exception as exc:  # noqa: BLE001 - framing must not fail the cut
            LOG.warning("active speaker detection failed (%s); following the most visible face", exc)
            speaking = {}

    for track in linked + tracks:
        track.crops = []

    path: list[PathSegment] = []
    turns: list[tuple[float, float, int, float | None]] = []
    min_turn = max(1, int(round(settings.min_turn_s * fps)))
    # Dead zone and speed are shares of the output width, i.e. of the window.
    deadzone_px = settings.deadzone * crop_w
    speed_px = settings.max_speed * crop_w
    for k, (a, b) in enumerate(shots):
        idx = np.array([i for i in range(n) if sample_shot[i] == k], dtype=int)
        if idx.size == 0:
            path.append(PathSegment(a, b, center_x, center_x))
            continue
        if k in split_ids:
            path.append(PathSegment(a, b, center_x, center_x))
            turns.append((a, b, -2, None))
            continue
        mine = [t for t in strong if t.shot == k]
        if not mine:
            path.append(PathSegment(a, b, center_x, center_x))
            turns.append((a, b, -1, None))
            continue

        series = {t.id: t.series(n) for t in mine}
        alive = np.zeros((len(mine) + 1, idx.size), dtype=bool)
        emissions = np.full((len(mine) + 1, idx.size), _NOBODY)
        for row, t in enumerate(mine):
            alive[row] = ~np.isnan(series[t.id][0][idx])
            if t.id in speaking:
                sc = speaking[t.id][idx]
                sc = np.where(np.isnan(sc), -_SCORE_CLIP, sc)
                sc = np.clip(_moving_average(sc, 3), -_SCORE_CLIP, _SCORE_CLIP)
            else:
                # No speaking evidence: prefer the face seen longest and largest.
                sc = np.full(idx.size, 1e-3 * len(t.idx) * float(np.median([f.height_px for f in t.faces])))
            emissions[row] = np.where(alive[row], sc, _DEAD)
        alive[-1] = True
        choice = choose_speakers(emissions, switch_penalty=settings.switch_penalty)
        choice = merge_short_turns(choice, alive, min_turn)

        camera = center_x
        runs = _runs(choice)
        for r, (state, i0, i1) in enumerate(runs):
            t0 = a if r == 0 else max(a, float(times[idx[i0]]) - settings.switch_lead_s)
            t1 = b if r == len(runs) - 1 else max(t0, float(times[idx[i1]]) - settings.switch_lead_s)
            if state == len(mine):
                path.append(PathSegment(t0, t1, camera, camera))
                turns.append((t0, t1, -1, None))
                continue
            track = mine[state]
            cx = series[track.id][0][idx[i0:i1]]
            valid = ~np.isnan(cx)
            seg_times = times[idx[i0:i1]][valid]
            xs = _crop_x(cx[valid], src_w, crop_w)
            if xs.size == 0:
                path.append(PathSegment(t0, t1, camera, camera))
                continue
            settle = xs[: max(1, int(round(fps)))]
            first = float(np.median(settle)) if settings.follow else float(np.median(xs))
            score = None
            if track.id in speaking:
                vals = speaking[track.id][idx[i0:i1]]
                vals = vals[~np.isnan(vals)]
                score = float(vals.mean()) if vals.size else None
            turns.append((t0, t1, track.id, score))
            if r > 0 and settings.switch == "pan" and abs(first - camera) <= 0.6 * crop_w:
                pan_end = min(t1, t0 + max(0.8, abs(first - camera) / max(speed_px, 1e-6)))
                path.append(PathSegment(t0, pan_end, camera, first))
                keep = seg_times >= pan_end
                seg_times, xs, t0 = seg_times[keep], xs[keep], pan_end
            if t1 - t0 <= 1e-3:
                camera = first
                continue
            planned = plan_follow(
                seg_times, xs, start_time=t0, start_value=first, end_time=t1,
                deadzone=deadzone_px, max_speed=speed_px, follow=settings.follow,
            )
            path.extend(planned)
            camera = planned[-1].v1 if planned else first

    path = [s for s in _merge_chained_glides(path) if s.t1 > s.t0]
    with_face = len({i for t in strong for i in t.idx})
    return FramingPlan(
        duration=duration,
        src_w=src_w,
        src_h=src_h,
        x_path=path or [PathSegment(0.0, duration, center_x, center_x)],
        split_shots=split_shots,
        rate=with_face / n,
        cuts=[a for a, _b in shots[1:]],
        turns=turns,
        tracks=[
            {"id": t.id, "shot": t.shot, "samples": len(t.idx), "cx": round(t.median_cx(), 3),
             "face_px": int(np.median([f.height_px for f in t.faces])),
             "start": round(t.first / fps, 2), "end": round(t.last / fps, 2)}
            for t in strong
        ],
        active_speaker_used=bool(speaking),
    )


def _split_panel_geometry(
    src_w: int, src_h: int, out_w: int, out_h: int, zoom: float = 0.9,
) -> tuple[int, int]:
    panel_h = out_h // 2
    aspect = out_w / panel_h
    crop_h = int(src_h * max(0.2, min(1.0, zoom)))
    crop_w = int(round(crop_h * aspect))
    if crop_w > src_w:
        crop_w = src_w
        crop_h = int(round(crop_w / aspect))
    return crop_w - crop_w % 2, crop_h - crop_h % 2


def center_plan(src_w: int, src_h: int, duration: float, *, out_w: int = 1080, out_h: int = 1920) -> FramingPlan:
    """A plan that just crops the middle (no faces, tracking off or unavailable)."""

    x = float(_crop_x(0.5, src_w, crop_width(src_w, src_h, out_w, out_h)))
    return FramingPlan(
        duration=duration, src_w=src_w, src_h=src_h,
        x_path=[PathSegment(0.0, duration, x, x)], split_shots=[], rate=0.0,
    )


def build_framing_filter(
    plan: FramingPlan, *, gpu: bool = False, out_w: int = 1080, out_h: int = 1920,
) -> str:
    """The -vf chain for a plan (input: decoded source frames).

    The window is cut from the source frame first — a cheap pointer move; the
    camera moves in steps of two source pixels (chroma alignment) — and only
    the window is scaled to ``out_w x out_h``. With ``gpu`` the scaling runs on
    the GPU (``hwupload_cuda`` + ``scale_cuda``) and the frame comes back
    only for the subtitle burn and NVENC; the input must then be decoded with
    ``-hwaccel cuda`` into system memory. Two-panel shots always use CPU
    filters (they are rare and small).
    """

    crop_w = crop_width(plan.src_w, plan.src_h, out_w, out_h)
    if crop_w > plan.src_w:
        # Narrower than 9:16: nothing to pan, fill the height and trim.
        return f"scale=w={out_w}:h={out_h}:force_original_aspect_ratio=increase,crop={out_w}:{out_h}"
    # No exact=1: crop then keeps x even. Decoded NV12 interleaves U and V in
    # one plane, and an odd x would start that plane on a V byte — every such
    # frame came out with swapped colours (an orange jumper turned blue).
    window = f"crop={crop_w}:{plan.src_h}:x='{piecewise_expr(plan.x_path)}':y=0"
    if not plan.split_shots:
        if gpu:
            return f"{window},hwupload_cuda,scale_cuda={out_w}:{out_h}:format=nv12,hwdownload,format=nv12"
        return f"{window},scale={out_w}:{out_h}"
    single = f"{window},scale={out_w}:{out_h}"
    if len(plan.split_shots) == 1:
        a, b, centers = plan.split_shots[0]
        if a <= 1e-3 and b >= plan.duration - 1e-3:
            return build_split_filter(src_w=plan.src_w, src_h=plan.src_h, centers=centers,
                                      target_w=out_w, target_h=out_h)

    crop_w, crop_h = _split_panel_geometry(plan.src_w, plan.src_h, out_w, out_h)
    panel_h = out_h // 2

    def panel_path(which: int, axis: int) -> str:
        segs: list[PathSegment] = []
        for a, b, centers in plan.split_shots:
            c = centers[which][axis]
            if axis == 0:
                v = min(max(c * plan.src_w - crop_w / 2.0, 0.0), plan.src_w - crop_w)
            else:
                v = min(max(c * plan.src_h - crop_h * 0.45, 0.0), plan.src_h - crop_h)
            segs.append(PathSegment(a, b, v, v))
        # Outside split shots the panels are hidden: hold the nearest value.
        filled: list[PathSegment] = []
        for k, seg in enumerate(segs):
            t0 = 0.0 if k == 0 else seg.t0
            t1 = plan.duration if k == len(segs) - 1 else segs[k + 1].t0
            filled.append(PathSegment(t0, t1, seg.v0, seg.v0))
        return piecewise_expr(filled, decimals=0)

    enable = "+".join(f"between(t,{a:.3f},{b - 1e-3:.3f})" for a, b, _c in plan.split_shots)
    return (
        "split=2[fs][fp];"
        f"[fs]{single}[fs1];"
        "[fp]split=2[t0][u0];"
        f"[t0]crop={crop_w}:{crop_h}:x='{panel_path(0, 0)}':y='{panel_path(0, 1)}',scale={out_w}:{panel_h}[t];"
        f"[u0]crop={crop_w}:{crop_h}:x='{panel_path(1, 0)}':y='{panel_path(1, 1)}',scale={out_w}:{panel_h}[u];"
        "[t][u]vstack=inputs=2[fp1];"
        f"[fs1][fp1]overlay=0:0:enable='{enable}'"
    )
