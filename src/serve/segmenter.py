"""
src/serve/segmenter.py — Continuous Signing Sentence Boundary Detector.

Implements an online sliding-window temporal motion & rest detector that segments
a continuous stream of landmark frames into sentence units without requiring
manual start/stop button clicks.

Boundary Detection Principles:
  1. Hand Activity: Sign language requires at least one hand active in signing space.
  2. Hands Dropping: Wrists drop below neutral chest area (y > wrist_drop_threshold)
     or disappear from frame.
  3. Low-Motion Pause: Velocity magnitude drops below rest threshold for consecutive frames.
  4. Temporal Duration Constraints:
     - Minimum length: reject brief micro-gestures / hesitations (< min_frames, e.g. 15 frames / 1s)
     - Maximum length: prevent buffer overflow for non-stop signing (> max_frames, e.g. 240 frames / 16s)
"""
from __future__ import annotations

import collections
import logging
from typing import Any

import numpy as np

logger = logging.getLogger("segmenter")


class ContinuousSignSegmenter:
    """
    Online sentence boundary detector for continuous ASL streaming.
    Processes frames one-by-one and detects when a complete sentence has been signed.
    """

    # Feature offsets in the 1200-dim feature vector
    LEFT_WRIST_Y_IDX = 61      # Pose point 15 y coordinate
    RIGHT_WRIST_Y_IDX = 65     # Pose point 16 y coordinate
    LEFT_HAND_MASK_IDX = 598   # Hand presence flag
    RIGHT_HAND_MASK_IDX = 599  # Hand presence flag
    VELOCITY_START_IDX = 600   # Velocity block start

    def __init__(
        self,
        min_frames: int = 15,          # Min frames for a sentence (~1.0s at 15 fps)
        max_frames: int = 240,         # Max frames before forcing a split (~16.0s at 15 fps)
        pause_frames: int = 12,        # Consecutive inactive/pause frames to trigger cut (~0.8s)
        motion_threshold: float = 0.5, # Velocity norm below which motion is considered paused
        wrist_drop_threshold: float = 1.8, # Shoulder-normalized Y threshold for dropped hands
        preroll_frames: int = 5,       # Frames to prepend before detected signing onset
    ):
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.pause_frames = pause_frames
        self.motion_threshold = motion_threshold
        self.wrist_drop_threshold = wrist_drop_threshold
        self.preroll_frames = preroll_frames

        # State tracking
        self.is_signing = False
        self.current_frames: list[np.ndarray] = []
        self.preroll_buffer: collections.deque[np.ndarray] = collections.deque(maxlen=preroll_frames)
        self.pause_counter: int = 0
        self.total_frames_processed: int = 0
        self.sentences_segmented: int = 0

    def reset(self) -> None:
        """Resets segmenter state."""
        self.is_signing = False
        self.current_frames.clear()
        self.preroll_buffer.clear()
        self.pause_counter = 0

    def is_frame_active(self, frame: np.ndarray) -> bool:
        """
        Determines whether the signer is actively signing in this frame:
        - Hand presence masks (at least one hand detected)
        - Wrist positions above the rest/drop line
        - Motion magnitude
        """
        lh_detected = frame[self.LEFT_HAND_MASK_IDX] > 0.5
        rh_detected = frame[self.RIGHT_HAND_MASK_IDX] > 0.5

        if not lh_detected and not rh_detected:
            return False

        # Check wrist elevations (lower y means higher elevation in image space)
        lw_y = frame[self.LEFT_WRIST_Y_IDX]
        rw_y = frame[self.RIGHT_WRIST_Y_IDX]

        hands_dropped = True
        if lh_detected and lw_y < self.wrist_drop_threshold:
            hands_dropped = False
        if rh_detected and rw_y < self.wrist_drop_threshold:
            hands_dropped = False

        if hands_dropped:
            return False

        # Check motion velocity
        vel_norm = float(np.linalg.norm(frame[self.VELOCITY_START_IDX:]))
        if vel_norm < self.motion_threshold:
            # Low motion holding still
            return False

        return True

    def process_frame(self, frame: np.ndarray) -> tuple[bool, np.ndarray | None, dict[str, Any]]:
        """
        Processes a single landmark frame (1200,).

        Returns:
            segment_ready (bool): True if a completed sentence segment was detected.
            segment_data (np.ndarray | None): Array of shape (T, 1200) representing the sentence.
            info (dict): State metadata (state, frame_count, vel_norm, pause_counter).
        """
        assert frame.shape == (1200,), f"Expected shape (1200,), got {frame.shape}"
        self.total_frames_processed += 1

        active = self.is_frame_active(frame)
        vel_norm = float(np.linalg.norm(frame[self.VELOCITY_START_IDX:]))

        segment_ready = False
        segment_data: np.ndarray | None = None
        cut_reason = ""

        if not self.is_signing:
            if active:
                # Transition: IDLE -> SIGNING
                self.is_signing = True
                self.current_frames = list(self.preroll_buffer)
                self.current_frames.append(frame)
                self.pause_counter = 0
            else:
                # Maintain circular pre-roll buffer during IDLE
                self.preroll_buffer.append(frame)
        else:
            # In SIGNING state
            self.current_frames.append(frame)

            if not active:
                self.pause_counter += 1
            else:
                self.pause_counter = 0

            # Check boundary conditions
            curr_len = len(self.current_frames)

            # Condition 1: Pause or hands dropped after sufficient length
            if self.pause_counter >= self.pause_frames:
                if curr_len >= self.min_frames:
                    segment_ready = True
                    cut_reason = "pause_detected"
                else:
                    # Too short: likely accidental twitch or brief hesitation, discard
                    self.reset()

            # Condition 2: Max duration reached
            elif curr_len >= self.max_frames:
                segment_ready = True
                cut_reason = "max_duration_reached"

            if segment_ready:
                # Remove trailing pause frames from the translated segment
                valid_len = max(self.min_frames, curr_len - self.pause_counter)
                segment_arr = np.array(self.current_frames[:valid_len], dtype=np.float32)
                segment_data = segment_arr
                self.sentences_segmented += 1

                # Reset for next sentence
                self.is_signing = False
                self.current_frames.clear()
                self.pause_counter = 0

        info = {
            "is_signing": self.is_signing,
            "buffered_frames": len(self.current_frames),
            "pause_counter": self.pause_counter,
            "vel_norm": round(vel_norm, 3),
            "cut_reason": cut_reason,
            "sentences_count": self.sentences_segmented,
        }

        return segment_ready, segment_data, info
