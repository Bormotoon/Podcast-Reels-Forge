"""RU: Кто из людей в кадре сейчас говорит (active speaker detection).

Модель — Light-ASD (Liao et al., CVPR 2023, MIT, github.com/Junhua-Liao/Light-ASD):
смотрит на губы (кроп лица 112x112, 25 к/с) и слушает звук (MFCC, 100 к/с)
одновременно, поэтому отличает говорящего от того, кто просто кивает или
поворачивает голову. Сеть маленькая (~1 млн параметров), веса качаются при
первом использовании и сверяются по sha256.

EN: Who of the people in frame is talking right now (active speaker detection).

The model is Light-ASD (Liao et al., CVPR 2023, MIT licence,
github.com/Junhua-Liao/Light-ASD): it watches the lips (112x112 face crops at
25 fps) and listens (MFCC at 100 fps) together, so it tells the speaker from
someone who merely nods or turns their head. The network is small (~1M
parameters); the weights are downloaded on first use and checked by sha256.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

LOG = logging.getLogger(__name__)

try:
    import torch
    from torch import nn

    HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is a core dependency
    HAS_TORCH = False

#: Video frame rate and MFCC frame rate the model was trained on.
VIDEO_FPS = 25
AUDIO_FPS = 100
SAMPLE_RATE = 16000
CROP_SIZE = 112

_MODEL_PATH = Path(__file__).resolve().parents[2] / "assets" / "models" / "light_asd_talkset.model"
MODEL_URL = (
    "https://raw.githubusercontent.com/Junhua-Liao/Light-ASD/"
    "ed38c232de5efe0261dbd68627c0ade7cdfe14eb/weight/finetuning_TalkSet.model"
)
MODEL_SHA256 = "efc375833887eefa9d209dc92810e18519b04c3c73ea35a549f2a7f40b7d94d5"

#: Frames per forward pass: bounds GPU memory (~100 MB) without hurting the
#: scores — the recurrent part sees 10 s of context either way.
_CHUNK_FRAMES = 250


# --------------------------------------------------------------------------
# MFCC, numerically the same as python_speech_features.mfcc (numcep=13,
# winlen=0.025, winstep=0.01, nfilt=26, nfft=512, preemph=0.97, ceplifter=22,
# appendEnergy=True) — what the model was trained on.
# --------------------------------------------------------------------------


def _mel_filterbank(nfilt: int, nfft: int, samplerate: int) -> np.ndarray:
    def hz2mel(hz: np.ndarray | float) -> np.ndarray:
        return 2595 * np.log10(1 + np.asarray(hz) / 700.0)

    def mel2hz(mel: np.ndarray) -> np.ndarray:
        return 700 * (10 ** (mel / 2595.0) - 1)

    melpoints = np.linspace(hz2mel(0.0), hz2mel(samplerate / 2), nfilt + 2)
    bins = np.floor((nfft + 1) * mel2hz(melpoints) / samplerate)
    fbank = np.zeros((nfilt, nfft // 2 + 1))
    for j in range(nfilt):
        for i in range(int(bins[j]), int(bins[j + 1])):
            fbank[j, i] = (i - bins[j]) / (bins[j + 1] - bins[j])
        for i in range(int(bins[j + 1]), int(bins[j + 2])):
            fbank[j, i] = (bins[j + 2] - i) / (bins[j + 2] - bins[j + 1])
    return fbank


def _dct_ortho_matrix(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    mat = np.cos(np.pi * k * (2 * i + 1) / (2 * n)) * np.sqrt(2.0 / n)
    mat[0] /= np.sqrt(2.0)
    return mat


def mfcc(signal: np.ndarray, samplerate: int = SAMPLE_RATE) -> np.ndarray:
    """13 MFCCs per 10 ms frame, shape (frames, 13)."""

    sig = np.asarray(signal, dtype=np.float64)
    if sig.size == 0:
        return np.zeros((0, 13))
    sig = np.append(sig[0], sig[1:] - 0.97 * sig[:-1])
    frame_len = int(round(0.025 * samplerate))
    frame_step = int(round(0.010 * samplerate))
    if sig.size <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(np.ceil((sig.size - frame_len) / frame_step))
    padlen = (numframes - 1) * frame_step + frame_len
    sig = np.concatenate((sig, np.zeros(padlen - sig.size)))
    idx = np.arange(frame_len)[None, :] + (np.arange(numframes) * frame_step)[:, None]
    frames = sig[idx]
    nfft = 512
    pspec = np.square(np.abs(np.fft.rfft(frames, nfft))) / nfft
    eps = np.finfo(float).eps
    energy = pspec.sum(axis=1)
    energy = np.where(energy == 0, eps, energy)
    feat = pspec @ _mel_filterbank(26, nfft, samplerate).T
    feat = np.where(feat == 0, eps, feat)
    feat = np.log(feat) @ _dct_ortho_matrix(26).T
    feat = feat[:, :13]
    lift = 1 + (22 / 2.0) * np.sin(np.pi * np.arange(13) / 22)
    feat = feat * lift
    feat[:, 0] = np.log(energy)
    return feat


# --------------------------------------------------------------------------
# Light-ASD network (layer names match the published state dict).
# --------------------------------------------------------------------------

if HAS_TORCH:

    class _AudioBlock(nn.Module):
        def __init__(self, cin: int, cout: int) -> None:
            super().__init__()
            self.relu = nn.ReLU()
            self.m_3 = nn.Conv2d(cin, cout, kernel_size=(3, 1), padding=(1, 0), bias=False)
            self.bn_m_3 = nn.BatchNorm2d(cout, momentum=0.01, eps=0.001)
            self.t_3 = nn.Conv2d(cout, cout, kernel_size=(1, 3), padding=(0, 1), bias=False)
            self.bn_t_3 = nn.BatchNorm2d(cout, momentum=0.01, eps=0.001)
            self.m_5 = nn.Conv2d(cin, cout, kernel_size=(5, 1), padding=(2, 0), bias=False)
            self.bn_m_5 = nn.BatchNorm2d(cout, momentum=0.01, eps=0.001)
            self.t_5 = nn.Conv2d(cout, cout, kernel_size=(1, 5), padding=(0, 2), bias=False)
            self.bn_t_5 = nn.BatchNorm2d(cout, momentum=0.01, eps=0.001)
            self.last = nn.Conv2d(cout, cout, kernel_size=(1, 1), bias=False)
            self.bn_last = nn.BatchNorm2d(cout, momentum=0.01, eps=0.001)

        def forward(self, x: Any) -> Any:
            x3 = self.relu(self.bn_t_3(self.t_3(self.relu(self.bn_m_3(self.m_3(x))))))
            x5 = self.relu(self.bn_t_5(self.t_5(self.relu(self.bn_m_5(self.m_5(x))))))
            return self.relu(self.bn_last(self.last(x3 + x5)))

    class _VisualBlock(nn.Module):
        def __init__(self, cin: int, cout: int, is_down: bool = False) -> None:
            super().__init__()
            self.relu = nn.ReLU()
            stride = (1, 2, 2) if is_down else (1, 1, 1)
            self.s_3 = nn.Conv3d(cin, cout, kernel_size=(1, 3, 3), stride=stride, padding=(0, 1, 1), bias=False)
            self.bn_s_3 = nn.BatchNorm3d(cout, momentum=0.01, eps=0.001)
            self.t_3 = nn.Conv3d(cout, cout, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
            self.bn_t_3 = nn.BatchNorm3d(cout, momentum=0.01, eps=0.001)
            self.s_5 = nn.Conv3d(cin, cout, kernel_size=(1, 5, 5), stride=stride, padding=(0, 2, 2), bias=False)
            self.bn_s_5 = nn.BatchNorm3d(cout, momentum=0.01, eps=0.001)
            self.t_5 = nn.Conv3d(cout, cout, kernel_size=(5, 1, 1), padding=(2, 0, 0), bias=False)
            self.bn_t_5 = nn.BatchNorm3d(cout, momentum=0.01, eps=0.001)
            self.last = nn.Conv3d(cout, cout, kernel_size=(1, 1, 1), bias=False)
            self.bn_last = nn.BatchNorm3d(cout, momentum=0.01, eps=0.001)

        def forward(self, x: Any) -> Any:
            x3 = self.relu(self.bn_t_3(self.t_3(self.relu(self.bn_s_3(self.s_3(x))))))
            x5 = self.relu(self.bn_t_5(self.t_5(self.relu(self.bn_s_5(self.s_5(x))))))
            return self.relu(self.bn_last(self.last(x3 + x5)))

    class _VisualEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block1 = _VisualBlock(1, 32, is_down=True)
            self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
            self.block2 = _VisualBlock(32, 64)
            self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
            self.block3 = _VisualBlock(64, 128)
            self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        def forward(self, x: Any) -> Any:
            x = self.block3(self.pool2(self.block2(self.pool1(self.block1(x)))))
            x = x.transpose(1, 2)
            b, t, c, w, h = x.shape
            x = self.maxpool(x.reshape(b * t, c, w, h))
            return x.view(b, t, c)

    class _AudioEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block1 = _AudioBlock(1, 32)
            self.pool1 = nn.MaxPool3d(kernel_size=(1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1))
            self.block2 = _AudioBlock(32, 64)
            self.pool2 = nn.MaxPool3d(kernel_size=(1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1))
            self.block3 = _AudioBlock(64, 128)

        def forward(self, x: Any) -> Any:
            x = self.block3(self.pool2(self.block2(self.pool1(self.block1(x)))))
            return torch.mean(x, dim=2, keepdim=True).squeeze(2).transpose(1, 2)

    class _BGRU(nn.Module):
        def __init__(self, channel: int) -> None:
            super().__init__()
            self.gru_forward = nn.GRU(channel, channel, num_layers=1, batch_first=True)
            self.gru_backward = nn.GRU(channel, channel, num_layers=1, batch_first=True)
            self.gelu = nn.GELU()

        def forward(self, x: Any) -> Any:
            x, _ = self.gru_forward(x)
            x = torch.flip(self.gelu(x), dims=[1])
            x, _ = self.gru_backward(x)
            return self.gelu(torch.flip(x, dims=[1]))

    class LightASD(nn.Module):
        """Audio-visual speaking score per video frame (logit; > 0 = speaking)."""

        def __init__(self) -> None:
            super().__init__()
            self.visualEncoder = _VisualEncoder()
            self.audioEncoder = _AudioEncoder()
            self.GRU = _BGRU(128)
            self.fc = nn.Linear(128, 2)

        def forward(self, audio: Any, video: Any) -> Any:
            b, t, w, h = video.shape
            v = (video.view(b, 1, t, w, h) / 255 - 0.4161) / 0.1688
            v = self.visualEncoder(v)
            a = self.audioEncoder(audio.unsqueeze(1).transpose(2, 3))
            n = min(a.shape[1], v.shape[1])
            x = self.GRU(a[:, :n] + v[:, :n])
            return self.fc(x.reshape(-1, 128))[:, 1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_asd_model(
    *, url: str = MODEL_URL, path: Path = _MODEL_PATH, sha256: str = MODEL_SHA256, timeout_s: int = 60,
) -> bool:
    """Download the Light-ASD weights if missing. False when unavailable."""

    if path.exists() and path.stat().st_size > 0:
        return True
    if os.environ.get("FORGE_NO_MODEL_DOWNLOAD") == "1":
        return False
    tmp = path.with_name(path.name + ".part")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=timeout_s) as response, tmp.open("wb") as out:  # noqa: S310
            out.write(response.read())
        if sha256 and _sha256(tmp) != sha256:
            LOG.warning("active speaker model from %s failed the checksum; not using it", url)
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(path)
        LOG.info("active speaker model downloaded to %s", path)
        return True
    except (OSError, ValueError) as exc:
        tmp.unlink(missing_ok=True)
        LOG.warning("active speaker model unavailable (%s): %s", url, exc)
        return False


def asd_available(*, download: bool = False) -> bool:
    if not HAS_TORCH:
        return False
    return _MODEL_PATH.exists() or (download and ensure_asd_model())


_LOCK = threading.Lock()
_MODELS: dict[str, Any] = {}


def _load(device: str) -> Any:
    with _LOCK:
        model = _MODELS.get(device)
        if model is None:
            if hasattr(torch.backends, "nnpack"):
                # Old CPUs: NNPACK prints a warning per convolution otherwise.
                torch.backends.nnpack.set_flags(False)
            state = torch.load(str(_MODEL_PATH), map_location="cpu", weights_only=True)
            model = LightASD()
            weights = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
            weights["fc.weight"] = state["lossAV.FC.weight"]
            weights["fc.bias"] = state["lossAV.FC.bias"]
            model.load_state_dict(weights)
            model.eval().to(device)
            _MODELS[device] = model
        return model


def speaking_scores(audio_feat: np.ndarray, crops: Any, first_frame: int, *, device: str = "cuda") -> np.ndarray:
    """Per-frame speaking logits for one face track.

    ``crops`` are the track's 112x112 grey face crops at 25 fps starting at
    clip frame ``first_frame`` (a uint8 tensor, ideally already on
    ``device``, or a numpy array); ``audio_feat`` is :func:`mfcc` of the whole
    clip (4 MFCC frames per video frame). Runs on ``device`` only: there is
    no silent CPU fallback.
    """

    total = int(crops.shape[0])
    if total == 0:
        return np.zeros(0)
    model = _load(device)
    video_all = crops if isinstance(crops, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(crops))
    audio_all = torch.from_numpy(np.ascontiguousarray(audio_feat[first_frame * 4 :], dtype=np.float32))
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, total, _CHUNK_FRAMES):
            n = min(_CHUNK_FRAMES, total - start, (audio_all.shape[0] - start * 4) // 4)
            if n <= 0:
                break
            video = video_all[start : start + n].to(device, dtype=torch.float32).unsqueeze(0)
            audio = audio_all[start * 4 : (start + n) * 4].to(device).unsqueeze(0)
            scores.append(model(audio, video).float().cpu().numpy())
    result = np.concatenate(scores) if scores else np.zeros(0)
    if result.shape[0] < total:
        # The audio ran out a few frames early: hold the last score.
        fill = result[-1] if result.size else 0.0
        result = np.concatenate([result, np.full(total - result.shape[0], fill)])
    return result


def face_crop_box(cx: float, cy: float, size: float, crop_scale: float = 0.4) -> tuple[float, float, float, float]:
    """Source rectangle (x0, y0, x1, y1) of the model's face crop.

    Light-ASD was trained on S3FD boxes padded by ``crop_scale``: the square
    reaches further below the face centre than above (chin and mouth in view).
    ``size`` is half the larger side of the face box.
    """

    return (
        cx - size * (1 + crop_scale),
        cy - size,
        cx + size * (1 + crop_scale),
        cy + size * (1 + 2 * crop_scale),
    )
