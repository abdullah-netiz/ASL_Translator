"""
Gate 4 Learning Curve Analysis: 25%, 50%, and 100% data scaling runs.

Evaluates whether model performance scales with increasing data volume
to inform the ~290GB dataset scaling decision (Gate 4).
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config, resolve_paths
from src.utils.seed import set_seed
from src.datasets.landmark_dataset import LandmarkDataset
from src.datasets.collate import pad_collate_fn
from src.datasets.augment import LandmarkAugmentor
from src.models.t5_translator import T5ASLTranslator
from src.train.trainer import Trainer

logger = logging.getLogger("learning_curve")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def run_learning_curve(
    fractions: list[float] = [0.25, 0.50, 1.0],
    epochs: int = 10,
    batch_size: int = 16,
    data_config: str = "configs/data.yaml",
    model_config: str = "configs/t5_decoder.yaml",
) -> dict:
    set_seed(42)
    root = Path(data_config).resolve().parents[1]
    data_cfg = resolve_paths(load_config(data_config), root)
    model_cfg = load_config(model_config)

    t5_model_name = model_cfg.get("t5_model", "t5-small")
    tokenizer = AutoTokenizer.from_pretrained(t5_model_name)

    train_manifest = Path(data_cfg["manifest_dir"]) / "train_manifest.csv"
    val_manifest = Path(data_cfg["manifest_dir"]) / "val_manifest.csv"

    full_train = LandmarkDataset(
        manifest_path=train_manifest,
        landmark_dir=data_cfg["landmark_dirs"]["train"],
        tokenizer=tokenizer,
        augmentor=LandmarkAugmentor(),
        max_seq_len=int(data_cfg.get("max_seq_len", 256)),
    )

    val_dataset = LandmarkDataset(
        manifest_path=val_manifest,
        landmark_dir=data_cfg["landmark_dirs"]["val"],
        tokenizer=tokenizer,
        augmentor=None,
        max_seq_len=int(data_cfg.get("max_seq_len", 256)),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn,
        num_workers=2,
    )

    results = []
    total_train_samples = len(full_train)
    logger.info(f"Available train samples: {total_train_samples}, val samples: {len(val_dataset)}")

    for frac in fractions:
        n_samples = max(4, int(round(total_train_samples * frac)))
        indices = list(range(total_train_samples))
        random.seed(42)
        random.shuffle(indices)
        selected_indices = indices[:n_samples]

        train_subset = Subset(full_train, selected_indices)
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=pad_collate_fn,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

        logger.info(f"\n{'='*50}\nTraining with {frac*100:.0f}% data ({n_samples} clips)...\n{'='*50}")

        model = T5ASLTranslator(
            t5_model_name=t5_model_name,
            in_features=1200,
            encoder_d_model=int(model_cfg.get("encoder_d_model", 256)),
            encoder_nhead=int(model_cfg.get("encoder_nhead", 8)),
            encoder_layers=int(model_cfg.get("encoder_layers", 4)),
            encoder_dropout=float(model_cfg.get("encoder_dropout", 0.1)),
        )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            tokenizer=tokenizer,
            checkpoint_dir=f"checkpoints/learning_curve_{int(frac*100)}",
            learning_rate=float(model_cfg.get("learning_rate", 3e-4)),
            warmup_steps=100,
            max_epochs=epochs,
        )

        fit_res = trainer.fit(eval_every=epochs)
        val_eval = trainer.evaluate(val_loader, beam_size=5, max_samples=50)
        metrics = val_eval["metrics"]

        entry = {
            "fraction": frac,
            "num_samples": n_samples,
            "train_loss": fit_res["history"][-1]["train_loss"],
            "val_loss": val_eval["loss"],
            "bleu": metrics.get("bleu", 0.0),
            "bleu_1": metrics.get("bleu_1", 0.0),
            "bleu_4": metrics.get("bleu_4", 0.0),
            "rouge_l": metrics.get("rouge_l", 0.0),
        }
        results.append(entry)
        logger.info(f"[{frac*100:.0f}% Data Result] val_loss={entry['val_loss']:.4f} | BLEU={entry['bleu']} | ROUGE-L={entry['rouge_l']}")

    # Decision logic for Gate 4 / 290G question
    bleu_scores = [r["bleu"] for r in results]
    improving = all(bleu_scores[i] <= bleu_scores[i+1] for i in range(len(bleu_scores)-1))
    recommendation = (
        "Scaling to ~290G dataset is JUSTIFIED (val performance scales monotonically with data)."
        if improving
        else "Scaling data NOT justified yet: gains have plateaued or gap is small. Focus on architecture, augmentation, or pretraining first."
    )

    summary = {
        "results": results,
        "recommendation": recommendation,
        "decision": "SCALE_DATA" if improving else "STAY_WITH_CURRENT_DATA",
    }

    out_path = Path("reports/learning_curve.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run learning curve analysis for Gate 4")
    parser.add_argument("--epochs", type=int, default=8, help="Epochs per run")
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.25, 0.50, 1.0])
    args = parser.parse_args()

    summary = run_learning_curve(fractions=args.fractions, epochs=args.epochs)
    print("\n" + "=" * 60)
    print("GATE 4 LEARNING CURVE SUMMARY")
    print("=" * 60)
    print(f"{'Fraction':<10} {'Clips':<10} {'Val Loss':<12} {'BLEU':<10} {'ROUGE-L':<10}")
    print("-" * 60)
    for r in summary["results"]:
        print(f"{r['fraction']*100:>7.0f}%   {r['num_samples']:<10} {r['val_loss']:<12.4f} {r['bleu']:<10.2f} {r['rouge_l']:<10.2f}")
    print("=" * 60)
    print("Recommendation:", summary["recommendation"])


if __name__ == "__main__":
    main()
