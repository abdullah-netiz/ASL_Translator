"""
CLI for evaluating ASL translation baseline and main models (LSTM, Transformer, T5) on validation or test sets.

Usage:
    python -m src.eval.evaluate --model lstm --checkpoint checkpoints/lstm/best_model.pt --split val
    python -m src.eval.evaluate --model transformer --checkpoint checkpoints/transformer/best_model.pt --split val
    python -m src.eval.evaluate --model t5 --checkpoint checkpoints/t5/best_model.pt --split val
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config, resolve_paths
from src.utils.logging import get_logger
from src.datasets.tokenizer import ASLTokenizer
from src.datasets.landmark_dataset import LandmarkDataset
from src.datasets.collate import pad_collate_fn
from src.models.lstm_seq2seq import LSTMSeq2Seq
from src.models.transformer_seq2seq import TransformerSeq2Seq
from src.models.t5_translator import T5ASLTranslator
from src.eval.metrics import compute_metrics, format_sample_table

logger = get_logger("evaluate")


def run_evaluation(
    model_type: str,
    checkpoint_path: str | Path,
    split: str = "val",
    data_config: str = "configs/data.yaml",
    beam_size: int = 5,
    max_samples: int = 100,
    device: str | None = None,
) -> dict[str, Any]:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_cfg = resolve_paths(load_config(data_config), Path(data_config).resolve().parents[1])

    is_t5 = model_type.lower() == "t5"
    if is_t5:
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
    else:
        tokenizer_path = Path("data/tokenizer/spm_bpe_8000.model")
        tokenizer = ASLTokenizer(tokenizer_path)

    manifest_path = Path(data_cfg["manifest_dir"]) / f"{split}_manifest.csv"
    dataset = LandmarkDataset(
        manifest_path=manifest_path,
        landmark_dir=data_cfg["landmark_dirs"][split],
        tokenizer=tokenizer,
        augmentor=None,
        max_seq_len=int(data_cfg.get("max_seq_len", 256)),
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=pad_collate_fn,
    )

    in_features = 1200
    vocab_size = getattr(tokenizer, "vocab_size", len(tokenizer))

    if is_t5:
        model: torch.nn.Module = T5ASLTranslator(
            t5_model_name="t5-small",
            in_features=in_features,
        )
    elif model_type.lower() == "lstm":
        model = LSTMSeq2Seq(
            in_features=in_features,
            vocab_size=vocab_size,
            pad_idx=tokenizer.pad_token_id,
            bos_idx=tokenizer.bos_token_id,
            eos_idx=tokenizer.eos_token_id,
        )
    elif model_type.lower() == "transformer":
        model = TransformerSeq2Seq(
            in_features=in_features,
            vocab_size=vocab_size,
            pad_idx=tokenizer.pad_token_id,
            bos_idx=tokenizer.bos_token_id,
            eos_idx=tokenizer.eos_token_id,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    ckpt_p = Path(checkpoint_path)
    if ckpt_p.exists():
        state = torch.load(ckpt_p, map_location=dev)
        model.load_state_dict(state.get("model_state_dict", state))
        logger.info(f"Loaded checkpoint from {ckpt_p}")
    else:
        logger.warning(f"Checkpoint {ckpt_p} does not exist; running with uninitialized model for testing.")

    model = model.to(dev).eval()

    clip_ids: list[str] = []
    references: list[str] = []
    hypotheses: list[str] = []

    with torch.no_grad():
        for batch in loader:
            landmarks = batch["landmarks"].to(dev)
            lm_mask = batch["landmarks_mask"].to(dev)

            preds = model.generate(landmarks, lm_mask, beam_size=beam_size)
            for pred_ids, ref_text, cid in zip(preds, batch["texts"], batch["clip_ids"]):
                hyp_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
                hypotheses.append(hyp_text)
                references.append(ref_text)
                clip_ids.append(cid)

            if len(hypotheses) >= max_samples:
                break

    metrics = compute_metrics(hypotheses, references)
    table = format_sample_table(clip_ids, hypotheses, references, num_samples=10)

    return {
        "metrics": metrics,
        "sample_table": table,
        "num_evaluated": len(hypotheses),
        "split": split,
        "model_type": model_type,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ASL translation baseline or main model")
    parser.add_argument("--model", required=True, choices=["lstm", "transformer", "t5"])
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()

    res = run_evaluation(
        model_type=args.model,
        checkpoint_path=args.checkpoint,
        split=args.split,
        beam_size=args.beam_size,
        max_samples=args.samples,
    )

    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS ({args.model.upper()} on {args.split.upper()} split)")
    print("=" * 60)
    for k, v in res["metrics"].items():
        print(f"  {k:<12}: {v}")
    print("\nSample Predictions:")
    print(res["sample_table"])


if __name__ == "__main__":
    main()
