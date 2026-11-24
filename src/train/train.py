"""
CLI for training ASL translation baseline and main models (LSTM, Transformer, T5).

Usage:
    python -m src.train.train --config configs/lstm.yaml --data-config configs/data.yaml
    python -m src.train.train --config configs/transformer.yaml --data-config configs/data.yaml
    python -m src.train.train --config configs/t5_decoder.yaml --data-config configs/data.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config, resolve_paths
from src.utils.logging import get_logger
from src.utils.seed import set_seed
from src.datasets.tokenizer import ASLTokenizer
from src.datasets.landmark_dataset import LandmarkDataset
from src.datasets.collate import pad_collate_fn
from src.datasets.augment import LandmarkAugmentor
from src.models.lstm_seq2seq import LSTMSeq2Seq
from src.models.transformer_seq2seq import TransformerSeq2Seq
from src.models.t5_translator import T5ASLTranslator
from src.train.trainer import Trainer

logger = get_logger("train")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ASL translation models")
    parser.add_argument("--config", required=True, help="Model config YAML (lstm.yaml, transformer.yaml, t5_decoder.yaml)")
    parser.add_argument("--data-config", default="configs/data.yaml", help="Data config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    args = parser.parse_args()

    root = Path(args.config).resolve().parents[1]
    model_cfg = load_config(args.config)
    data_cfg = resolve_paths(load_config(args.data_config), root)

    seed = int(model_cfg.get("seed", 42))
    set_seed(seed)

    config_name = Path(args.config).stem.lower()

    # 1. Load Tokenizer
    is_t5 = "t5" in config_name
    if is_t5:
        t5_model_name = model_cfg.get("t5_model", "t5-small")
        logger.info(f"Loading pretrained HuggingFace tokenizer: {t5_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
    else:
        tok_path = Path("data/tokenizer/spm_bpe_8000.model")
        if not tok_path.exists():
            raise FileNotFoundError(f"SentencePiece tokenizer model missing at {tok_path}. Train it first.")
        tokenizer = ASLTokenizer(model_path=tok_path)

    # 2. Datasets
    augmentor = LandmarkAugmentor()
    train_manifest = Path(data_cfg["manifest_dir"]) / "train_manifest.csv"
    val_manifest = Path(data_cfg["manifest_dir"]) / "val_manifest.csv"

    train_dataset = LandmarkDataset(
        manifest_path=train_manifest,
        landmark_dir=data_cfg["landmark_dirs"]["train"],
        tokenizer=tokenizer,
        augmentor=augmentor,
        max_seq_len=int(data_cfg.get("max_seq_len", 256)),
    )

    val_dataset = LandmarkDataset(
        manifest_path=val_manifest,
        landmark_dir=data_cfg["landmark_dirs"]["val"],
        tokenizer=tokenizer,
        augmentor=None,
        max_seq_len=int(data_cfg.get("max_seq_len", 256)),
    )

    logger.info(f"Loaded datasets: {len(train_dataset)} train samples, {len(val_dataset)} val samples")

    batch_size = args.batch_size or int(model_cfg.get("batch_size", 32))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate_fn,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn,
        num_workers=2,
    )

    # 3. Model selection
    in_features = 1200
    vocab_size = getattr(tokenizer, "vocab_size", len(tokenizer))

    if is_t5:
        t5_model_name = model_cfg.get("t5_model", "t5-small")
        model = T5ASLTranslator(
            t5_model_name=t5_model_name,
            in_features=in_features,
            encoder_d_model=int(model_cfg.get("encoder_d_model", 256)),
            encoder_nhead=int(model_cfg.get("encoder_nhead", 8)),
            encoder_layers=int(model_cfg.get("encoder_layers", 4)),
            encoder_dropout=float(model_cfg.get("encoder_dropout", 0.1)),
        )
        ckpt_dir = Path("checkpoints/t5")
    elif "lstm" in config_name:
        model = LSTMSeq2Seq(
            in_features=in_features,
            vocab_size=vocab_size,
            d_model=int(model_cfg.get("d_model", 256)),
            hidden_size=int(model_cfg.get("hidden_size", 512)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.3)),
            pad_idx=tokenizer.pad_token_id,
            bos_idx=tokenizer.bos_token_id,
            eos_idx=tokenizer.eos_token_id,
        )
        ckpt_dir = Path("checkpoints/lstm")
    elif "transformer" in config_name:
        model = TransformerSeq2Seq(
            in_features=in_features,
            vocab_size=vocab_size,
            d_model=int(model_cfg.get("d_model", 256)),
            nhead=int(model_cfg.get("nhead", 8)),
            num_encoder_layers=int(model_cfg.get("num_encoder_layers", 4)),
            num_decoder_layers=int(model_cfg.get("num_decoder_layers", 4)),
            dim_feedforward=int(model_cfg.get("dim_feedforward", 1024)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            pad_idx=tokenizer.pad_token_id,
            bos_idx=tokenizer.bos_token_id,
            eos_idx=tokenizer.eos_token_id,
        )
        ckpt_dir = Path("checkpoints/transformer")
    else:
        raise ValueError(f"Unsupported config type: {config_name}")

    # 4. Trainer
    epochs = args.epochs or int(model_cfg.get("max_epochs", 50))
    lr = float(model_cfg.get("learning_rate", 3e-4))
    warmup_steps = int(model_cfg.get("warmup_steps", 500))

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        checkpoint_dir=ckpt_dir,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        max_epochs=epochs,
        device=args.device,
    )

    fit_res = trainer.fit(eval_every=5)
    logger.info(f"Training complete. Best val loss: {fit_res['best_val_loss']:.4f}")

    # 5. Final evaluation with beam search (width 5) on best checkpoint
    best_ckpt_path = ckpt_dir / "best_model.pt"
    if best_ckpt_path.exists():
        logger.info(f"Reloading best checkpoint from {best_ckpt_path} for final evaluation...")
        best_state = torch.load(best_ckpt_path, map_location=trainer.device)
        trainer.model.load_state_dict(best_state["model_state_dict"])
    logger.info("Running final beam search (width 5) evaluation on validation set...")
    val_eval = trainer.evaluate(val_loader, beam_size=5, max_samples=50)
    metrics = val_eval["metrics"]
    logger.info(f"Final Validation Metrics: {metrics}")


if __name__ == "__main__":
    main()
