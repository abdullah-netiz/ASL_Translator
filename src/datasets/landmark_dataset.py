"""
PyTorch Dataset for ASL landmark sequences and English translation targets.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class LandmarkDataset(Dataset):
    """
    Loads pre-extracted float16 .npy landmark files and text translation pairs.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        landmark_dir: str | Path,
        tokenizer: Any = None,
        augmentor: Callable[[np.ndarray], np.ndarray] | None = None,
        max_seq_len: int = 256,
        max_target_len: int = 128,
        require_landmark_file: bool = True,
    ):
        self.manifest_path = Path(manifest_path)
        self.landmark_dir = Path(landmark_dir)
        self.tokenizer = tokenizer
        self.augmentor = augmentor
        self.max_seq_len = max_seq_len
        self.max_target_len = max_target_len

        self.samples: list[dict[str, Any]] = []
        self._load_manifest(require_landmark_file)

    def _load_manifest(self, require_file: bool) -> None:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with open(self.manifest_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                clip_id = row.get("clip_id", "").strip()
                sentence = row.get("sentence", "").strip()
                lm_path = self.landmark_dir / f"{clip_id}.npy"

                if require_file and not lm_path.exists():
                    continue

                self.samples.append({
                    "clip_id": clip_id,
                    "sentence": sentence,
                    "lm_path": str(lm_path),
                })

        logger.info(
            f"Loaded {len(self.samples)} samples from {self.manifest_path.name} "
            f"(landmark_dir: {self.landmark_dir.name})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        lm_path = item["lm_path"]

        # Load numpy landmarks
        if Path(lm_path).exists():
            arr = np.load(lm_path)  # (T, F) float16
        else:
            raise FileNotFoundError(f"Landmark file missing: {lm_path}")

        # Apply augmentation if configured
        if self.augmentor is not None:
            arr = self.augmentor(arr)

        # Truncate to max_seq_len
        if len(arr) > self.max_seq_len:
            arr = arr[:self.max_seq_len]

        landmarks = torch.from_numpy(arr.astype(np.float32))

        # Tokenize sentence
        tokens: torch.Tensor
        if self.tokenizer is not None:
            if hasattr(self.tokenizer, "pad_token_id") and not hasattr(self.tokenizer, "sp"):
                # Hugging Face tokenizer (T5)
                token_ids = self.tokenizer.encode(item["sentence"], max_length=self.max_target_len, truncation=True)
            else:
                # ASLTokenizer (SentencePiece)
                token_ids = self.tokenizer.encode(item["sentence"], add_bos=True, add_eos=True)
                if len(token_ids) > self.max_target_len:
                    token_ids = token_ids[:self.max_target_len - 1] + [self.tokenizer.eos_token_id]
            tokens = torch.tensor(token_ids, dtype=torch.long)
        else:
            tokens = torch.empty(0, dtype=torch.long)

        return {
            "landmarks": landmarks,
            "tokens": tokens,
            "clip_id": item["clip_id"],
            "text": item["sentence"],
            "length": len(landmarks),
        }
