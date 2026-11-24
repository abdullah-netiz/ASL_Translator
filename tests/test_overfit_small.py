"""
Gate 2 Sanity Check: Overfit on ~50 clips until loss approaches zero.

Verifies:
  1. The full data pipeline (Dataset, Tokenizer, Collate) works seamlessly with the model.
  2. Gradients propagate properly through the entire architecture.
  3. The model has sufficient capacity to memorize a small set of real How2Sign clips.
"""
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.datasets.landmark_dataset import LandmarkDataset
from src.datasets.collate import pad_collate_fn
from src.datasets.tokenizer import ASLTokenizer
from src.models.lstm_seq2seq import LSTMSeq2Seq


def test_overfit_small_dataset():
    manifest_path = Path("data/manifests/train_manifest.csv")
    landmark_dir = Path("data/landmarks/train")
    tokenizer_path = Path("data/tokenizer/spm_bpe_8000.model")

    if not manifest_path.exists() or not landmark_dir.exists() or not tokenizer_path.exists():
        pytest.skip("Required data files for Gate 2 overfit test not found.")

    # 1. Load tokenizer
    tokenizer = ASLTokenizer(model_path=tokenizer_path)

    # 2. Load dataset
    full_dataset = LandmarkDataset(
        manifest_path=manifest_path,
        landmark_dir=landmark_dir,
        tokenizer=tokenizer,
        max_seq_len=150,
        max_target_len=64,
        require_landmark_file=True,
    )

    num_samples = min(50, len(full_dataset))
    if num_samples < 5:
        pytest.skip(f"Not enough extracted clips for overfit test (found {num_samples}).")

    dataset = Subset(full_dataset, range(num_samples))
    print(f"\n[Gate 2] Running overfit test on {num_samples} real clips...")

    # 3. Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=pad_collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4. Instantiate model
    model = LSTMSeq2Seq(
        in_features=1200,
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        hidden_size=512,
        num_layers=2,
        dropout=0.0,  # No dropout for memorization test
        pad_idx=tokenizer.pad_token_id,
        bos_idx=tokenizer.bos_token_id,
        eos_idx=tokenizer.eos_token_id,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 5. Overfit loop
    epochs = 60
    initial_loss = None
    final_loss = None

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_tokens = 0

        for batch in loader:
            landmarks = batch["landmarks"].to(device)
            lm_mask = batch["landmarks_mask"].to(device)
            tokens = batch["tokens"].to(device)
            labels = batch["target_labels"].to(device)

            optimizer.zero_grad()
            # Predict tokens 1..L given 0..L-1
            logits = model(landmarks, lm_mask, tokens, teacher_forcing_ratio=1.0)

            # logits: (B, L-1, V), targets: (B, L-1)
            targets = labels[:, 1:].contiguous()
            loss = criterion(logits.view(-1, tokenizer.vocab_size), targets.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            non_pad = (targets != -100).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad

        epoch_loss = total_loss / max(1, total_tokens)
        if initial_loss is None:
            initial_loss = epoch_loss
        final_loss = epoch_loss

        if epoch % 10 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:2d}/{epochs} | Loss: {epoch_loss:.4f}")

    print(f"\n[Gate 2] Initial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f}")

    # Gate 2 assertion: loss drops dramatically towards zero
    assert final_loss < 0.25, f"Overfit failed: final loss {final_loss:.4f} is too high (expected < 0.25)."
    assert final_loss < initial_loss * 0.1, f"Loss did not decrease sufficiently: {initial_loss:.4f} -> {final_loss:.4f}"
    print("Gate 2 PASS: Model successfully memorized 50 clips!")
