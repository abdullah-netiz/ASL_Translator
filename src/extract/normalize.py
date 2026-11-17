"""
Landmark normalization pipeline.

Input : list of per-frame raw landmark dicts (from HolisticExtractor)
        + landmarks.yaml config dict
Output: np.ndarray of shape (T, F) in float16

Feature layout per frame (concatenated):
  static block:
    pose_norm        : (n_pose, 4)  shoulder-centered, shoulder-width-scaled
    lh_norm          : (21, 4)      same coordinate space
    rh_norm          : (21, 4)
    face_norm        : (n_face, 4)
    lh_wrist_rel     : (21, 3)      x,y,z relative to hand wrist (landmark 0)
    rh_wrist_rel     : (21, 3)
    hand_masks       : (2,)         [lh_detected, rh_detected] ∈ {0, 1}
  velocity block     : same size as static block (frame[t] - frame[t-1]; zero at t=0)

Total F = 2 * static_dim
where static_dim = n_pose*4 + 21*4 + 21*4 + n_face*4 + 21*3 + 21*3 + 2
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _shoulder_mid(pose: np.ndarray, l_idx: int, r_idx: int) -> np.ndarray:
    """Midpoint between left and right shoulder (xyz)."""
    return (pose[l_idx, :3] + pose[r_idx, :3]) / 2.0


def _shoulder_width(pose: np.ndarray, l_idx: int, r_idx: int) -> float:
    """Euclidean shoulder width; floored at 1e-6 to prevent div-by-zero."""
    return float(np.linalg.norm(pose[l_idx, :3] - pose[r_idx, :3])) + 1e-6


def _shoulder_normalize(pts: np.ndarray, mid: np.ndarray, scale: float) -> np.ndarray:
    """Translate pts[:, :3] to shoulder-origin and scale by shoulder width."""
    out = pts.copy()
    out[:, :3] = (pts[:, :3] - mid) / scale
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_sequence(
    frames: list[dict],
    lm_cfg: dict,
    max_seq_len: int = 256,
) -> np.ndarray:
    """
    Normalize a list of raw frame dicts into a float16 feature array.

    Parameters
    ----------
    frames      : output of HolisticExtractor.extract_video()
    lm_cfg      : loaded configs/landmarks.yaml
    max_seq_len : clip sequences longer than this are truncated

    Returns
    -------
    np.ndarray of shape (T, F), dtype float16
    """
    pose_indices: list[int] = lm_cfg["pose_indices"]
    face_indices: list[int] = lm_cfg["face_indices"]
    l_sh: int = lm_cfg["left_shoulder_idx"]
    r_sh: int = lm_cfg["right_shoulder_idx"]

    # Cap length before any processing
    frames = frames[:max_seq_len]
    T = len(frames)

    static_vecs: list[np.ndarray] = []

    for fr in frames:
        pose_full: np.ndarray = fr["pose"]        # (33, 4)
        lh_full:   np.ndarray = fr["left_hand"]   # (21, 4)
        rh_full:   np.ndarray = fr["right_hand"]  # (21, 4)
        face_full: np.ndarray = fr["face"]        # (468|478, 4)
        lh_det = float(fr["left_hand_detected"])
        rh_det = float(fr["right_hand_detected"])

        mid   = _shoulder_mid(pose_full, l_sh, r_sh)
        scale = _shoulder_width(pose_full, l_sh, r_sh)

        # Pose subset — shoulder-normalized
        pose_sub  = pose_full[pose_indices, :]     # (n_pose, 4)
        pose_norm = _shoulder_normalize(pose_sub, mid, scale)

        # Hands — shoulder-normalized
        lh_norm = _shoulder_normalize(lh_full, mid, scale)  # (21, 4)
        rh_norm = _shoulder_normalize(rh_full, mid, scale)  # (21, 4)

        # Wrist-relative hand streams (extra hand-shape representation)
        # landmark 0 of each hand = wrist in shoulder-normalized space
        lh_wrist_rel = lh_norm[:, :3] - lh_norm[0, :3]   # (21, 3)
        rh_wrist_rel = rh_norm[:, :3] - rh_norm[0, :3]   # (21, 3)

        # Face subset — shoulder-normalized
        # Guard against face arrays shorter than the highest face index
        if face_full.shape[0] > max(face_indices):
            face_sub = face_full[face_indices, :]
        else:
            # Fallback: pad to required size with zeros
            padded = np.zeros((max(face_indices) + 1, 4), dtype=np.float32)
            padded[: face_full.shape[0]] = face_full
            face_sub = padded[face_indices, :]
        face_norm = _shoulder_normalize(face_sub, mid, scale)  # (n_face, 4)

        vec = np.concatenate([
            pose_norm.ravel(),       # n_pose * 4
            lh_norm.ravel(),         # 21 * 4
            rh_norm.ravel(),         # 21 * 4
            face_norm.ravel(),       # n_face * 4
            lh_wrist_rel.ravel(),    # 21 * 3
            rh_wrist_rel.ravel(),    # 21 * 3
            np.array([lh_det, rh_det], dtype=np.float32),  # 2
        ], dtype=np.float32)
        static_vecs.append(vec)

    static = np.stack(static_vecs, axis=0)   # (T, F_static)

    # Velocity: Δ between consecutive frames; first frame = zeros
    velocity = np.zeros_like(static)
    if T > 1:
        velocity[1:] = static[1:] - static[:-1]

    result = np.concatenate([static, velocity], axis=1)  # (T, 2*F_static)
    return result.astype(np.float16)


def feature_dim(lm_cfg: dict) -> int:
    """Return the total feature dimension F for a given landmarks config."""
    n_pose = len(lm_cfg["pose_indices"])
    n_face = len(lm_cfg["face_indices"])
    static_dim = (
        n_pose * 4   # pose x,y,z,vis
        + 21 * 4     # left hand
        + 21 * 4     # right hand
        + n_face * 4 # face subset
        + 21 * 3     # lh wrist-relative xyz
        + 21 * 3     # rh wrist-relative xyz
        + 2          # hand masks
    )
    return 2 * static_dim  # static + velocity
