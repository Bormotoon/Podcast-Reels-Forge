"""RU: Раскладка кадра по найденным лицам. EN: Frame layout from detected faces."""

from __future__ import annotations

import re
from pathlib import Path

from podcast_reels_forge.utils.face_crop import (
    Face,
    build_split_filter,
    decide_layout,
    ensure_face_model,
    merge_faces,
)


def _face(cx: float, size: float = 120.0, cy: float = 0.4) -> Face:
    return Face(cx=cx, cy=cy, width_px=size, height_px=size)


def test_one_person_gives_a_single_crop() -> None:
    layout = decide_layout([[_face(0.30)], [_face(0.32)], [_face(0.31)], []])
    assert layout.kind == "single"
    assert layout.primary is not None and abs(layout.primary[0] - 0.31) < 1e-9
    assert layout.rate == 0.75


def test_two_steady_people_are_stacked() -> None:
    samples = [[_face(0.25), _face(0.75)] for _ in range(5)] + [[_face(0.25)]]
    layout = decide_layout(samples)
    assert layout.kind == "split"
    assert [round(c[0], 2) for c in layout.centers] == [0.25, 0.75]


def test_an_occasional_second_face_does_not_split() -> None:
    samples = [[_face(0.3)] for _ in range(5)] + [[_face(0.3), _face(0.8)]]
    assert decide_layout(samples).kind == "single"


def test_two_faces_side_by_side_are_not_two_speakers() -> None:
    samples = [[_face(0.45), _face(0.55)] for _ in range(4)]
    assert decide_layout(samples).kind == "single"


def test_no_faces() -> None:
    layout = decide_layout([[], []])
    assert layout.kind == "none" and layout.primary is None and layout.rate == 0.0


def test_duplicates_from_tiles_are_merged() -> None:
    merged = merge_faces([_face(0.30, size=100), _face(0.31, size=140), _face(0.7)])
    assert sorted(round(f.cx, 2) for f in merged) == [0.31, 0.7]


def test_split_filter_stacks_two_panels_inside_the_frame() -> None:
    graph = build_split_filter(src_w=1920, src_h=1080, centers=[(0.25, 0.4), (0.75, 0.4)])
    crops = [tuple(map(int, m)) for m in re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", graph)]
    assert len(crops) == 2
    for w, h, x, y in crops:
        assert abs(w / h - 1080 / 960) < 0.01
        assert 0 <= x <= 1920 - w and 0 <= y <= 1080 - h
    assert crops[0][2] < crops[1][2], "left speaker on top"
    assert graph.endswith("vstack=inputs=2")
    assert "scale=1080:960" in graph


def test_model_is_downloaded_once(tmp_path: Path) -> None:
    source = tmp_path / "model.tflite"
    source.write_bytes(b"tflite")
    target = tmp_path / "models" / "blaze.tflite"
    assert ensure_face_model(url=source.as_uri(), path=str(target))
    assert target.read_bytes() == b"tflite"
    source.unlink()
    assert ensure_face_model(url=source.as_uri(), path=str(target)), "already there"


def test_a_failed_download_is_not_fatal(tmp_path: Path) -> None:
    missing = tmp_path / "nowhere.tflite"
    assert not ensure_face_model(url=missing.as_uri(), path=str(tmp_path / "m.tflite"))
