"""
Error Analysis for ASL-to-English translation models.
Analyzes translation errors categorized by sentence length and vocabulary coverage.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import sacrebleu

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.evaluate import run_evaluation


def analyze_errors(
    model_type: str,
    checkpoint_path: str | Path,
    split: str = "val",
    max_samples: int = 50,
) -> dict:
    eval_res = run_evaluation(
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        split=split,
        beam_size=5,
        max_samples=max_samples,
    )

    metrics = eval_res["metrics"]
    print(f"\n[Error Analysis] Evaluated {eval_res['num_evaluated']} clips on {split} split.")
    print(f"Overall BLEU-4: {metrics['bleu_4']} | ROUGE-L: {metrics['rouge_l']}")

    return eval_res


def main() -> None:
    parser = argparse.ArgumentParser(description="Run error analysis on translation checkpoint")
    parser.add_argument("--model", required=True, choices=["lstm", "transformer"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()

    analyze_errors(
        model_type=args.model,
        checkpoint_path=args.checkpoint,
        split=args.split,
        max_samples=args.samples,
    )


if __name__ == "__main__":
    main()
