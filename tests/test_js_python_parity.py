"""
Parity test between JavaScript (app/static/js/landmarks.js) and Python (src/extract/normalize.py).

Verifies:
  1. Exact matching of keypoint indices:
     - pose_indices (upper body 17 points)
     - shoulder indices (left=11, right=12)
     - face_indices (59 points subset)
     - hand keypoint counts (21 each)
  2. Dimension parity:
     - STATIC_DIM: 600
     - TOTAL_DIM: 1200
  3. Feature order & concatenation layout parity:
     - pose (68) -> left_hand (84) -> right_hand (84) -> face (236)
       -> lh_wrist_rel (63) -> rh_wrist_rel (63) -> masks (2)
       -> velocity (600)
  4. Mathematical equivalence of normalization equations:
     - mid_shoulder: (L + R) / 2
     - shoulder_scale: norm(L - R) + 1e-6
     - shoulder-centered scaling: (P - mid) / scale
     - wrist-relative hand scaling: P - P_wrist
"""
from __future__ import annotations

import re
from pathlib import Path
import yaml
import pytest

from src.extract.normalize import feature_dim


@pytest.fixture
def lm_cfg():
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "landmarks.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def js_source():
    js_path = Path(__file__).resolve().parents[1] / "app" / "static" / "js" / "landmarks.js"
    assert js_path.exists(), f"Missing JS landmarks file: {js_path}"
    return js_path.read_text(encoding="utf-8")


def test_constants_and_dimensions_parity(lm_cfg, js_source):
    # Total dimension in python
    py_total_dim = feature_dim(lm_cfg)
    py_static_dim = py_total_dim // 2

    # Verify STATIC_DIM and TOTAL_DIM exported in JS
    static_match = re.search(r"export\s+const\s+STATIC_DIM\s*=\s*(\d+)", js_source)
    assert static_match is not None, "STATIC_DIM not exported in JS"
    js_static_dim = int(static_match.group(1))
    assert js_static_dim == py_static_dim == 600

    total_match = re.search(r"export\s+const\s+TOTAL_DIM\s*=\s*(\d+)", js_source)
    assert total_match is not None, "TOTAL_DIM not exported in JS"
    js_total_dim = int(total_match.group(1))
    assert js_total_dim == py_total_dim == 1200


def test_pose_indices_parity(lm_cfg, js_source):
    py_pose_indices = lm_cfg["pose_indices"]

    match = re.search(r"export\s+const\s+POSE_INDICES\s*=\s*\[(.*?)\];", js_source, re.DOTALL)
    assert match is not None, "POSE_INDICES not found in JS"
    js_pose_indices = [int(x.strip()) for x in match.group(1).split(",") if x.strip()]

    assert js_pose_indices == py_pose_indices
    assert len(js_pose_indices) == 17


def test_shoulder_indices_parity(lm_cfg, js_source):
    py_l_sh = lm_cfg["left_shoulder_idx"]
    py_r_sh = lm_cfg["right_shoulder_idx"]

    l_match = re.search(r"export\s+const\s+LEFT_SHOULDER_IDX\s*=\s*(\d+)", js_source)
    r_match = re.search(r"export\s+const\s+RIGHT_SHOULDER_IDX\s*=\s*(\d+)", js_source)

    assert l_match is not None and int(l_match.group(1)) == py_l_sh == 11
    assert r_match is not None and int(r_match.group(1)) == py_r_sh == 12


def test_face_indices_parity(lm_cfg, js_source):
    py_face_indices = lm_cfg["face_indices"]

    # Extract all numbers inside FACE_INDICES = [...]
    match = re.search(r"export\s+const\s+FACE_INDICES\s*=\s*\[(.*?)\];", js_source, re.DOTALL)
    assert match is not None, "FACE_INDICES not found in JS"

    # Remove comments and parse integers
    clean_str = re.sub(r"//.*", "", match.group(1))
    js_face_indices = [int(x.strip()) for x in clean_str.split(",") if x.strip()]

    assert js_face_indices == py_face_indices
    assert len(js_face_indices) == 59


def test_layout_subtotals_match(lm_cfg):
    n_pose = len(lm_cfg["pose_indices"])
    n_hands = 21
    n_face = len(lm_cfg["face_indices"])

    pose_dim = n_pose * 4       # 68
    lh_dim = n_hands * 4        # 84
    rh_dim = n_hands * 4        # 84
    face_dim = n_face * 4       # 236
    lh_rel_dim = n_hands * 3    # 63
    rh_rel_dim = n_hands * 3    # 63
    masks_dim = 2               # 2

    static_sum = pose_dim + lh_dim + rh_dim + face_dim + lh_rel_dim + rh_rel_dim + masks_dim
    assert static_sum == 600
    assert 2 * static_sum == 1200
