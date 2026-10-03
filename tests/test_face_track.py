"""RU: Слежение за говорящим. EN: Following whoever is talking."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

from podcast_reels_forge.utils import active_speaker
from podcast_reels_forge.utils.face_crop import Face, _MODEL_PATH
from podcast_reels_forge.utils.face_track import (
    FramingPlan,
    PathSegment,
    build_framing_filter,
    build_tracks,
    center_plan,
    choose_speakers,
    crop_grid,
    crop_width,
    is_still_picture,
    merge_short_turns,
    path_value,
    piecewise_expr,
    plan_follow,
    scene_cuts,
    shots_from_cuts,
    split_at_shots,
)


def _eval_expr(expr: str, t: float) -> float:
    """Evaluate an ffmpeg expression of ``t`` with the few functions we emit."""

    py = expr.replace("PI", "math.pi")
    py = re.sub(r"\bgte\(", "_gte(", py)
    py = re.sub(r"\blt\(", "_lt(", py)
    env = {"math": math, "cos": math.cos, "t": t,
           "_gte": lambda a, b: float(a >= b), "_lt": lambda a, b: float(a < b)}
    return float(eval(py, env))  # noqa: S307 - test-only, our own string


def test_shots_ignore_flashes_and_edges() -> None:
    shots = shots_from_cuts([0.3, 5.0, 5.4, 9.8], 10.0, 1.0)
    assert shots == [(0.0, 5.0), (5.0, 10.0)]


def test_scene_cut_is_a_jump_not_steady_motion() -> None:
    steady = [6.0] * 20
    assert scene_cuts(np.array(steady), 10.0, fps=25) == []
    jump = np.array([1.0] * 10 + [40.0] + [1.0] * 10)
    assert scene_cuts(jump, 10.0, fps=25) == [pytest.approx(11 / 25)]


def test_viterbi_ignores_a_short_interjection() -> None:
    a = np.full(50, 3.0)
    b = np.full(50, -3.0)
    b[20:22] = 3.0  # 0.4 s "uh-huh" while A keeps talking a bit quieter
    a[20:22] = 0.0
    nobody = np.full(50, -4.0)
    path = choose_speakers(np.stack([a, b, nobody]), switch_penalty=10.0)
    assert set(path.tolist()) == {0}


def test_viterbi_switches_on_a_real_turn() -> None:
    a = np.concatenate([np.full(25, 3.0), np.full(25, -3.0)])
    b = -a
    nobody = np.full(50, -4.0)
    path = choose_speakers(np.stack([a, b, nobody]), switch_penalty=10.0)
    assert path[0] == 0 and path[-1] == 1
    switch = int(np.argmax(path == 1))
    assert 23 <= switch <= 27


def test_dead_tracks_are_never_chosen() -> None:
    a = np.full(10, 3.0)
    a[5:] = -1e9  # left the frame
    b = np.full(10, -2.0)
    nobody = np.full(10, -4.0)
    path = choose_speakers(np.stack([a, b, nobody]), switch_penalty=10.0)
    assert path[:5].tolist() == [0] * 5 and path[5:].tolist() == [1] * 5


def test_short_turns_go_to_a_neighbour() -> None:
    path = np.array([0] * 10 + [1] * 2 + [0] * 10)
    alive = np.ones((2, path.size), dtype=bool)
    assert set(merge_short_turns(path, alive, 5).tolist()) == {0}


def test_camera_holds_inside_the_dead_zone() -> None:
    times = np.arange(50) * 0.2
    xs = 500 + 20 * np.sin(np.arange(50))  # fidgeting in place
    segs = plan_follow(times, xs, start_time=0.0, start_value=500.0, end_time=10.0,
                       deadzone=60.0, max_speed=300.0, follow=True)
    assert len(segs) == 1 and segs[0].v0 == segs[0].v1 == 500.0


def test_camera_glides_to_where_the_person_stops() -> None:
    times = np.arange(60) * 0.2
    xs = np.concatenate([np.full(15, 300.0), np.linspace(300, 700, 10), np.full(35, 700.0)])
    segs = plan_follow(times, xs, start_time=0.0, start_value=300.0, end_time=12.0,
                       deadzone=60.0, max_speed=300.0, follow=True)
    glides = [s for s in segs if s.v0 != s.v1]
    assert len(glides) == 1
    assert glides[0].v1 == pytest.approx(700.0)
    assert segs[0].t0 == 0.0 and segs[-1].t1 == 12.0
    # Contiguous, and the value never jumps between segments.
    for prev, nxt in zip(segs, segs[1:]):
        assert prev.t1 == pytest.approx(nxt.t0)
        assert prev.v1 == pytest.approx(nxt.v0)


def test_no_move_right_before_the_turn_ends() -> None:
    times = np.arange(25) * 0.2
    xs = np.concatenate([np.full(21, 300.0), np.full(4, 700.0)])  # moves at 4.2 s of 5
    segs = plan_follow(times, xs, start_time=0.0, start_value=300.0, end_time=5.0,
                       deadzone=60.0, max_speed=300.0, follow=True)
    assert all(s.v0 == s.v1 == 300.0 for s in segs)


def test_follow_off_holds_one_framing() -> None:
    times = np.arange(10) * 0.2
    xs = np.linspace(100, 900, 10)
    segs = plan_follow(times, xs, start_time=0.0, start_value=400.0, end_time=2.0,
                       deadzone=10.0, max_speed=300.0, follow=False)
    assert segs == [PathSegment(0.0, 2.0, 400.0, 400.0)]


def test_expression_matches_the_planned_path() -> None:
    segs = [
        PathSegment(0.0, 2.0, 100.0, 100.0),
        PathSegment(2.0, 3.0, 100.0, 400.0),
        PathSegment(3.0, 5.0, 400.0, 400.0),
        PathSegment(5.0, 8.0, 50.0, 50.0),  # a cut
    ]
    expr = piecewise_expr(segs)
    for t in (0.0, 1.0, 2.0, 2.25, 2.5, 2.99, 3.0, 4.9, 5.0, 7.0, 9.0):
        assert _eval_expr(expr, t) == pytest.approx(path_value(segs, t), abs=0.2)
    assert _eval_expr(expr, 2.5) == pytest.approx(250.0, abs=0.2), "ease-in-out midpoint"


def _face(cx: float, cy: float = 0.3, h: float = 100.0) -> Face:
    return Face(cx=cx, cy=cy, width_px=h * 0.8, height_px=h)


def test_tracks_keep_identities_and_never_cross_a_cut() -> None:
    rng = np.random.default_rng(1)
    samples = [[_face(0.3 + rng.normal(0, 0.003)), _face(0.7 + rng.normal(0, 0.003))] for _ in range(20)]
    tracks = build_tracks(samples, [0] * 10 + [1] * 10, src_w=1920, src_h=1080, fps=5)
    assert len(tracks) == 4
    assert all(len(t.idx) == 10 for t in tracks)
    linked = build_tracks(samples, [0] * 20, src_w=1920, src_h=1080, fps=5)
    pieces = split_at_shots(linked, [0] * 10 + [1] * 10, 5)
    assert sorted(len(t.idx) for t in pieces) == [10, 10, 10, 10]


def test_a_face_on_a_slide_is_a_still_picture() -> None:
    rng = np.random.default_rng(2)
    photo = build_tracks([[_face(0.5 + rng.normal(0, 0.0002))] for _ in range(40)], [0] * 40,
                         src_w=1920, src_h=1080, fps=5)[0]
    person = build_tracks([[_face(0.5 + rng.normal(0, 0.004), 0.3 + rng.normal(0, 0.004))] for _ in range(40)],
                          [0] * 40, src_w=1920, src_h=1080, fps=5)[0]
    assert is_still_picture(photo, src_w=1920, src_h=1080)
    assert not is_still_picture(person, src_w=1920, src_h=1080)


def test_gpu_filter_scales_on_the_gpu_and_keeps_chroma_aligned() -> None:
    plan = FramingPlan(duration=10.0, src_w=1920, src_h=1080,
                       x_path=[PathSegment(0.0, 5.0, 100.0, 100.0), PathSegment(5.0, 10.0, 900.0, 900.0)],
                       split_shots=[], rate=1.0)
    gpu = build_framing_filter(plan, gpu=True)
    assert gpu.startswith(f"crop={crop_width(1920, 1080)}:1080:x='")
    assert "hwupload_cuda,scale_cuda=1080:1920" in gpu and gpu.endswith("hwdownload,format=nv12")
    # exact=1 would cut NV12 at odd x and swap U and V.
    assert "exact" not in gpu
    cpu = build_framing_filter(plan, gpu=False)
    assert cpu.endswith("scale=1080:1920") and "cuda" not in cpu


def test_center_plan_and_narrow_sources() -> None:
    plan = center_plan(1920, 1080, 30.0)
    assert plan.x_path[0].v0 == pytest.approx((1920 - crop_width(1920, 1080)) / 2)
    narrow = center_plan(600, 1280, 30.0)  # narrower than 9:16: nothing to pan
    assert "force_original_aspect_ratio=increase" in build_framing_filter(narrow, gpu=True)


def test_crop_grid_matches_the_reference_crop() -> None:
    """grid_sample gives the crop Light-ASD was trained on (resize to 224, keep the centre)."""

    torch = pytest.importorskip("torch")
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(3)
    gray = (rng.random((240, 320)) * 255).astype(np.uint8)
    gray = cv2.GaussianBlur(gray, (7, 7), 2)
    box = active_speaker.face_crop_box(150.0, 100.0, 30.0)
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    ref = cv2.resize(gray[y0:y1, x0:x1], (224, 224))[56:168, 56:168].astype(float)
    grid = crop_grid(torch.tensor([[x0, y0, x1, y1]], dtype=torch.float32), 320, 240)
    out = torch.nn.functional.grid_sample(
        torch.from_numpy(gray).float()[None, None], grid, mode="bilinear", align_corners=False,
    )[0, 0].numpy()
    assert np.abs(out - ref).mean() < 1.5


def test_mfcc_shape_and_energy_column() -> None:
    feats = active_speaker.mfcc(np.zeros(16000, dtype=np.int16) + 100)
    assert feats.shape == (99, 13)
    loud = active_speaker.mfcc((np.sin(np.arange(16000) / 5) * 8000).astype(np.int16))
    assert loud[:, 0].mean() > feats[:, 0].mean(), "first column is log energy"


@pytest.mark.skipif(not Path(_MODEL_PATH).exists(), reason="YuNet model not downloaded")
def test_torch_yunet_matches_opencv() -> None:
    """The torch rebuild of YuNet gives OpenCV's raw outputs on the same input."""

    torch = pytest.importorskip("torch")
    cv2 = pytest.importorskip("cv2")
    from podcast_reels_forge.utils.face_crop import _build_yunet

    rng = np.random.default_rng(4)
    image = (rng.random((1, 3, 160, 224)) * 255).astype(np.float32)
    net = cv2.dnn.readNetFromONNX(_MODEL_PATH)
    net.setInput(image)
    names = [f"{kind}_{s}" for s in (8, 16, 32) for kind in ("cls", "obj", "bbox", "kps")]
    ref = dict(zip(names, net.forward(names)))
    model = _build_yunet("cpu")
    with torch.no_grad():
        heads = model(torch.from_numpy(image))
    for stride, (cls, obj, bbox, kps) in zip((8, 16, 32), heads):
        np.testing.assert_allclose(torch.sigmoid(cls).numpy().reshape(-1),
                                   ref[f"cls_{stride}"].reshape(-1), atol=1e-4)
        np.testing.assert_allclose(torch.sigmoid(obj).numpy().reshape(-1),
                                   ref[f"obj_{stride}"].reshape(-1), atol=1e-4)
        np.testing.assert_allclose(bbox.numpy().reshape(-1), ref[f"bbox_{stride}"].reshape(-1), atol=1e-3)
        np.testing.assert_allclose(kps.numpy().reshape(-1), ref[f"kps_{stride}"].reshape(-1), atol=1e-3)
