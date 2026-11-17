"""
Landmark-space data augmentation for ASL sequences.

Rules (per instructions.md):
- Small 2D rotation (in xy plane), scale jitter, translation jitter, Gaussian noise
- Random frame dropout
- Mild speed change / temporal resampling (0.9x - 1.1x)
- STRICT RULE: Do NOT horizontally mirror (swaps handedness and breaks signs)
"""
from __future__ import annotations

import math
import random
from typing import Sequence

import numpy as np
import torch


class LandmarkAugmentor:
    """
    Augments landmark sequences of shape (T, F) or (T, 1200).
    """

    def __init__(
        self,
        rotation_degrees: float = 10.0,
        scale_range: tuple[float, float] = (0.9, 1.1),
        translation_std: float = 0.02,
        noise_std: float = 0.005,
        frame_dropout_prob: float = 0.08,
        speed_range: tuple[float, float] = (0.9, 1.1),
        static_dim: int = 600,
    ):
        self.rotation_degrees = rotation_degrees
        self.scale_range = scale_range
        self.translation_std = translation_std
        self.noise_std = noise_std
        self.frame_dropout_prob = frame_dropout_prob
        self.speed_range = speed_range
        self.static_dim = static_dim

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply augmentations to sequence x of shape (T, F).
        Returns augmented sequence of shape (T_aug, F) as float32 or float16.
        """
        if len(x) < 4:
            return x

        orig_dtype = x.dtype
        # Convert to float32 for augmentation math
        arr = x.astype(np.float32)

        # 1. Mild temporal speed change (resampling)
        if self.speed_range and self.speed_range[0] < self.speed_range[1]:
            arr = self._speed_change(arr)

        # 2. Random frame dropout
        if self.frame_dropout_prob > 0.0:
            arr = self._frame_dropout(arr)

        # 3. Spatial transformations on static block
        static = arr[:, :self.static_dim].copy()

        # Scale jitter
        if self.scale_range:
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            static = self._apply_scale(static, scale)

        # Translation jitter
        if self.translation_std > 0.0:
            static = self._apply_translation(static)

        # 2D in-plane rotation
        if self.rotation_degrees > 0.0:
            angle_rad = math.radians(random.uniform(-self.rotation_degrees, self.rotation_degrees))
            static = self._apply_rotation(static, angle_rad)

        # Gaussian noise
        if self.noise_std > 0.0:
            noise = np.random.normal(0.0, self.noise_std, size=static.shape).astype(np.float32)
            # Do not add noise to mask flags (last 2 dimensions of static block)
            noise[:, -2:] = 0.0
            static += noise

        # Recompute velocity block for consistency
        velocity = np.zeros_like(static)
        if len(static) > 1:
            velocity[1:] = static[1:] - static[:-1]

        out = np.concatenate([static, velocity], axis=-1)
        return out.astype(orig_dtype)

    def _speed_change(self, arr: np.ndarray) -> np.ndarray:
        speed = random.uniform(self.speed_range[0], self.speed_range[1])
        T, F = arr.shape
        new_T = max(4, int(round(T / speed)))
        if new_T == T:
            return arr

        old_idx = np.linspace(0, T - 1, T)
        new_idx = np.linspace(0, T - 1, new_T)
        resampled = np.empty((new_T, F), dtype=arr.dtype)
        for d in range(F):
            resampled[:, d] = np.interp(new_idx, old_idx, arr[:, d])
        return resampled

    def _frame_dropout(self, arr: np.ndarray) -> np.ndarray:
        T = len(arr)
        keep_mask = np.random.rand(T) > self.frame_dropout_prob
        # Ensure at least 4 frames are kept
        if np.sum(keep_mask) < 4:
            return arr
        return arr[keep_mask]

    def _apply_scale(self, static: np.ndarray, scale: float) -> np.ndarray:
        out = static.copy()
        # Scale all coordinates (leave mask flags unchanged)
        out[:, :-2] *= scale
        return out

    def _apply_translation(self, static: np.ndarray) -> np.ndarray:
        out = static.copy()
        dx = random.gauss(0.0, self.translation_std)
        dy = random.gauss(0.0, self.translation_std)
        dz = random.gauss(0.0, self.translation_std)
        # Apply to x, y, z channels
        for i in range(0, self.static_dim - 2, 4):
            if i + 3 <= self.static_dim - 2:
                out[:, i] += dx
                out[:, i + 1] += dy
                out[:, i + 2] += dz
        return out

    def _apply_rotation(self, static: np.ndarray, angle_rad: float) -> np.ndarray:
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        out = static.copy()
        # Rotate x, y coordinates
        for i in range(0, self.static_dim - 2, 4):
            x = static[:, i]
            y = static[:, i + 1]
            out[:, i] = x * cos_a - y * sin_a
            out[:, i + 1] = x * sin_a + y * cos_a
        return out
