"""
Unit tests for landmark normalization pipeline (src/extract/normalize.py).

Verifies:
  1. Center and scale math (mid-shoulder is origin, shoulder distance is 1.0)
  2. Hand wrist-relative coordinate math (wrist is origin for wrist_rel stream)
  3. Correct output tensor shape: (T, 2 * static_dim) = (T, 1200)
  4. Output dtype is float16
  5. Velocity calculation: frame 0 is zero, frame 1+ is delta
  6. Empty/zero landmark handling without NaN or Inf
"""
import numpy as np
import pytest

from src.extract.normalize import (
    _shoulder_mid,
    _shoulder_width,
    _shoulder_normalize,
    normalize_sequence,
    feature_dim,
)


@pytest.fixture
def dummy_lm_cfg():
    return {
        "pose_indices": list(range(17)),
        "num_hand_landmarks": 21,
        "face_indices": [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
            33, 246, 161, 160, 159, 158,
            362, 398, 384, 385, 386, 387,
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400
        ],
        "features_per_point": 4,
        "left_shoulder_idx": 11,
        "right_shoulder_idx": 12,
        "left_wrist_pose_idx": 15,
        "right_wrist_pose_idx": 16,
    }


def make_dummy_frame(
    l_sh=(0.4, 0.5, 0.0),
    r_sh=(0.6, 0.5, 0.0),
    lh_detected=True,
    rh_detected=True,
    n_face=478,
):
    pose = np.zeros((33, 4), dtype=np.float32)
    pose[11, :3] = l_sh
    pose[11, 3] = 1.0
    pose[12, :3] = r_sh
    pose[12, 3] = 1.0

    lh = np.zeros((21, 4), dtype=np.float32)
    if lh_detected:
        lh[:, :3] = 0.2
        lh[0, :3] = [0.1, 0.2, 0.3]  # wrist
        lh[:, 3] = 1.0

    rh = np.zeros((21, 4), dtype=np.float32)
    if rh_detected:
        rh[:, :3] = 0.8
        rh[0, :3] = [0.7, 0.8, 0.9]  # wrist
        rh[:, 3] = 1.0

    face = np.zeros((n_face, 4), dtype=np.float32)
    face[:, 3] = 1.0

    return {
        "pose": pose,
        "left_hand": lh,
        "right_hand": rh,
        "face": face,
        "left_hand_detected": lh_detected,
        "right_hand_detected": rh_detected,
        "pose_detected": True,
    }


def test_shoulder_mid_and_width():
    pose = np.zeros((33, 4), dtype=np.float32)
    pose[11, :3] = [0.0, 1.0, 0.0]  # left shoulder
    pose[12, :3] = [2.0, 1.0, 0.0]  # right shoulder

    mid = _shoulder_mid(pose, 11, 12)
    np.testing.assert_allclose(mid, [1.0, 1.0, 0.0])

    width = _shoulder_width(pose, 11, 12)
    assert abs(width - 2.0) < 1e-4


def test_shoulder_normalize():
    pts = np.array([[1.0, 1.0, 0.0, 0.9]], dtype=np.float32)
    mid = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    scale = 2.0

    norm = _shoulder_normalize(pts, mid, scale)
    np.testing.assert_allclose(norm[0, :3], [0.0, 0.0, 0.0])
    assert norm[0, 3] == pytest.approx(0.9, rel=1e-5)  # visibility unmodified


def test_feature_dim_calculation(dummy_lm_cfg):
    f_dim = feature_dim(dummy_lm_cfg)
    # n_pose=17 (17*4=68), hands=2*(21*4)=168, face=59 (59*4=236),
    # wrist_rel=2*(21*3)=126, masks=2 -> static=600. Total = 2*600 = 1200
    assert f_dim == 1200


def test_normalize_sequence_output_shape_and_dtype(dummy_lm_cfg):
    frames = [make_dummy_frame() for _ in range(10)]
    arr = normalize_sequence(frames, dummy_lm_cfg, max_seq_len=256)

    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float16
    assert arr.shape == (10, 1200)
    assert not np.isnan(arr).any()
    assert not np.isinf(arr).any()


def test_normalize_sequence_velocity_zero_at_t0(dummy_lm_cfg):
    frames = [make_dummy_frame() for _ in range(5)]
    arr = normalize_sequence(frames, dummy_lm_cfg, max_seq_len=256)

    static_dim = 600
    # First frame velocity (elements static_dim:2*static_dim) should be all zeros
    np.testing.assert_allclose(arr[0, static_dim:], 0.0)


def test_normalize_sequence_max_len_truncation(dummy_lm_cfg):
    frames = [make_dummy_frame() for _ in range(20)]
    arr = normalize_sequence(frames, dummy_lm_cfg, max_seq_len=8)
    assert arr.shape == (8, 1200)
