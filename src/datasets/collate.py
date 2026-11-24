"""
Collate functions and batch samplers for LandmarkDataset.
Handles padding, attention masks, label preparation, and length bucketing.
"""
from __future__ import annotations

import random
from typing import Any, Iterator, Sequence

import torch
from torch.utils.data import Sampler


def pad_collate_fn(
    batch: list[dict[str, Any]],
    pad_token_id: int = 0,
    ignore_index: int = -100,
) -> dict[str, Any]:
    """
    Pad variable-length landmark sequences and token sequences.

    Returns dict with keys:
        landmarks       : (B, T_max, F) float32 tensor
        landmarks_mask  : (B, T_max) bool tensor (True for real frames, False for pad)
        tokens          : (B, L_max) long tensor padded with pad_token_id
        tokens_mask     : (B, L_max) bool tensor (True for real tokens, False for pad)
        target_labels   : (B, L_max) long tensor (pad_token_id replaced with ignore_index)
        clip_ids        : list[str]
        texts           : list[str]
    """
    batch_size = len(batch)
    if batch_size == 0:
        raise ValueError("Cannot collate an empty batch.")

    f_dim = batch[0]["landmarks"].shape[-1]
    max_t = max(len(b["landmarks"]) for b in batch)
    max_l = max(len(b["tokens"]) for b in batch) if len(batch[0]["tokens"]) > 0 else 0

    # Padded landmarks: (B, T_max, F)
    padded_lm = torch.zeros((batch_size, max_t, f_dim), dtype=torch.float32)
    lm_mask = torch.zeros((batch_size, max_t), dtype=torch.bool)

    for i, b in enumerate(batch):
        t = len(b["landmarks"])
        padded_lm[i, :t, :] = b["landmarks"]
        lm_mask[i, :t] = True

    # Padded tokens: (B, L_max)
    padded_tokens = torch.full((batch_size, max_l), fill_value=pad_token_id, dtype=torch.long)
    tokens_mask = torch.zeros((batch_size, max_l), dtype=torch.bool)
    target_labels = torch.full((batch_size, max_l), fill_value=ignore_index, dtype=torch.long)

    if max_l > 0:
        for i, b in enumerate(batch):
            l = len(b["tokens"])
            padded_tokens[i, :l] = b["tokens"]
            tokens_mask[i, :l] = True
            target_labels[i, :l] = b["tokens"]

    return {
        "landmarks": padded_lm,
        "landmarks_mask": lm_mask,
        "tokens": padded_tokens,
        "tokens_mask": tokens_mask,
        "target_labels": target_labels,
        "clip_ids": [b["clip_id"] for b in batch],
        "texts": [b["text"] for b in batch],
    }


class LengthBucketBatchSampler(Sampler[list[int]]):
    """
    Sorts samples by sequence length and yields mini-batches of similar lengths
    to minimize padding overhead during training.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        shuffle: bool = True,
        bucket_size_multiplier: int = 20,
    ):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket_size = batch_size * bucket_size_multiplier

    def __iter__(self) -> Iterator[list[int]]:
        indices = list(range(len(self.lengths)))

        if self.shuffle:
            random.shuffle(indices)

        # Process in mega-buckets
        for i in range(0, len(indices), self.bucket_size):
            bucket = indices[i:i + self.bucket_size]
            # Sort within mega-bucket by length
            bucket.sort(key=lambda idx: self.lengths[idx])

            # Form mini-batches from sorted bucket
            batches = [
                bucket[j:j + self.batch_size]
                for j in range(0, len(bucket), self.batch_size)
            ]

            if self.shuffle:
                random.shuffle(batches)

            for b in batches:
                yield b

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size
