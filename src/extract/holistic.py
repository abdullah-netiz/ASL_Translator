"""MediaPipe Holistic landmark extractor for a single video clip.

Wraps the MediaPipe 0.10.14 Holistic (legacy) API — NOT the Tasks API.
Pin mediapipe==0.10.14 in requirements.txt; version drift silently changes outputs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# mediapipe 0.10.14 — Holistic legacy API
try:
    import mediapipe as mp
    _mp_holistic = mp.solutions.holistic
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class HolisticExtractor:
    """
    Wraps MediaPipe Holistic (0.10.14) for frame-level landmark extraction.

    Returns per-frame structured dicts; normalization is done in normalize.py.
    """

    def __init__(self, static_image_mode: bool = False, model_complexity: int = 1):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "mediapipe is not installed. Run: pip install mediapipe==0.10.14"
            )
        if not CV2_AVAILABLE:
            raise ImportError(
                "opencv-python is not installed. Run: pip install opencv-python==4.10.0.84"
            )
        self.holistic = _mp_holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=False,
            refine_face_landmarks=True,
        )

    def process_frame(self, bgr_frame: np.ndarray) -> dict[str, Any]:
        """
        Process a single BGR frame.

        Returns a dict with keys:
            pose            : np.ndarray (33, 4)  x,y,z,visibility; zeros if absent
            left_hand       : np.ndarray (21, 4)
            right_hand      : np.ndarray (21, 4)
            face            : np.ndarray (468|478, 4) full mesh; caller slices by face_indices
            left_hand_detected  : bool
            right_hand_detected : bool
            pose_detected       : bool
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        result = self.holistic.process(rgb)

        def _lm_to_array(lm_list, n: int) -> tuple[np.ndarray, bool]:
            if lm_list is None:
                return np.zeros((n, 4), dtype=np.float32), False
            arr = np.array(
                [
                    [lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0)]
                    for lm in lm_list.landmark
                ],
                dtype=np.float32,
            )
            return arr, True

        pose_arr, pose_ok = _lm_to_array(result.pose_landmarks, 33)
        lh_arr, lh_ok = _lm_to_array(result.left_hand_landmarks, 21)
        rh_arr, rh_ok = _lm_to_array(result.right_hand_landmarks, 21)

        # refine_face_landmarks=True gives 478 points; without it 468
        n_face = (
            478
            if result.face_landmarks and len(result.face_landmarks.landmark) == 478
            else 468
        )
        face_arr, _ = _lm_to_array(result.face_landmarks, n_face)

        return {
            "pose": pose_arr,
            "left_hand": lh_arr,
            "right_hand": rh_arr,
            "face": face_arr,
            "left_hand_detected": lh_ok,
            "right_hand_detected": rh_ok,
            "pose_detected": pose_ok,
        }

    def extract_video(
        self,
        video_path: str | Path,
        stride: int = 2,
        max_frames: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Extract landmarks from every `stride`-th frame.

        Parameters
        ----------
        video_path : path to .mp4 clip
        stride     : keep every n-th frame (stride=2 ≈ 15 fps from 30 fps source)
        max_frames : hard cap on output frames (applied before normalization)

        Returns
        -------
        List of per-frame dicts (see process_frame).
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        frames: list[dict] = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride == 0:
                frames.append(self.process_frame(frame))
                if max_frames and len(frames) >= max_frames:
                    break
            frame_idx += 1
        cap.release()
        return frames

    def close(self) -> None:
        self.holistic.close()

    def __enter__(self) -> "HolisticExtractor":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
