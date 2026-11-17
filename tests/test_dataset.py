"""
Unit tests for LandmarkDataset, Collate, and Augmentor.
"""
import tempfile
import csv
from pathlib import Path
import numpy as np
import torch
import pytest

from src.datasets.landmark_dataset import LandmarkDataset
from src.datasets.collate import pad_collate_fn, LengthBucketBatchSampler
from src.datasets.augment import LandmarkAugmentor
from src.datasets.tokenizer import ASLTokenizer


@pytest.fixture
def dummy_data_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        lm_dir = tmp_path / "landmarks"
        lm_dir.mkdir()

        # Create 5 dummy .npy landmark files
        manifest_rows = []
        for i in range(5):
            clip_id = f"clip_{i}"
            length = 10 + i * 5
            arr = np.random.randn(length, 1200).astype(np.float16)
            np.save(lm_dir / f"{clip_id}.npy", arr)
            manifest_rows.append({
                "clip_id": clip_id,
                "video_path": f"/dummy/{clip_id}.mp4",
                "sentence": f"This is sentence {i} for testing.",
                "duration_start": "0.0",
                "duration_end": "5.0",
                "split": "train",
            })

        manifest_file = tmp_path / "manifest.csv"
        with open(manifest_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["clip_id", "video_path", "sentence", "duration_start", "duration_end", "split"])
            writer.writeheader()
            writer.writerows(manifest_rows)

        yield manifest_file, lm_dir


def test_dataset_loading(dummy_data_dir):
    manifest_file, lm_dir = dummy_data_dir
    dataset = LandmarkDataset(
        manifest_path=manifest_file,
        landmark_dir=lm_dir,
        max_seq_len=50,
    )

    assert len(dataset) == 5
    sample = dataset[0]
    assert "landmarks" in sample
    assert "tokens" in sample
    assert "clip_id" in sample
    assert sample["landmarks"].shape == (10, 1200)
    assert sample["landmarks"].dtype == torch.float32


def test_collate_fn(dummy_data_dir):
    manifest_file, lm_dir = dummy_data_dir
    dataset = LandmarkDataset(
        manifest_path=manifest_file,
        landmark_dir=lm_dir,
    )

    batch_samples = [dataset[0], dataset[1], dataset[2]]
    batch = pad_collate_fn(batch_samples, pad_token_id=0, ignore_index=-100)

    assert "landmarks" in batch
    assert "landmarks_mask" in batch
    assert "tokens" in batch
    assert "tokens_mask" in batch
    assert "target_labels" in batch

    # Check padded shape: B=3, max length of [10, 15, 20] is 20
    assert batch["landmarks"].shape == (3, 20, 1200)
    assert batch["landmarks_mask"].shape == (3, 20)
    assert batch["landmarks_mask"][0, 10:].sum() == 0  # padded part is False
    assert batch["landmarks_mask"][0, :10].all()      # valid part is True


def test_length_bucket_batch_sampler():
    lengths = [10, 50, 12, 48, 15, 52, 11, 49]
    sampler = LengthBucketBatchSampler(lengths, batch_size=2, shuffle=False)
    batches = list(sampler)
    assert len(batches) == 4
    # All indices should be covered
    all_indices = [idx for b in batches for idx in b]
    assert sorted(all_indices) == list(range(8))


def test_landmark_augmentor():
    augmentor = LandmarkAugmentor(
        rotation_degrees=10.0,
        scale_range=(0.95, 1.05),
        translation_std=0.01,
        noise_std=0.005,
        frame_dropout_prob=0.1,
        speed_range=(0.95, 1.05),
    )

    x = np.random.randn(30, 1200).astype(np.float16)
    x_aug = augmentor(x)

    assert isinstance(x_aug, np.ndarray)
    assert x_aug.shape[-1] == 1200
    assert x_aug.dtype == np.float16
    assert not np.isnan(x_aug).any()
