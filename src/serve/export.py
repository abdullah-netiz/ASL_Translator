"""
src/serve/export.py — Export ASL Translation Models to TorchScript and ONNX.

Exports the Landmark Encoder + Learned Projection pipeline to TorchScript (.pt)
and ONNX (.onnx) for production deployment, and benchmarks eager vs exported latency.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.models.t5_translator import T5ASLTranslator
from src.serve.inference import ASLInferencePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger("export")


class ExportableLandmarkEncoder(nn.Module):
    """
    Submodule containing the LandmarkEncoder and Projection layer,
    suitable for TorchScript tracing and ONNX export.
    Inputs:
      - landmarks: Tensor of shape (B, T, in_features)
      - landmarks_mask: Tensor of shape (B, T) bool
    Outputs:
      - projected_features: Tensor of shape (B, T, d_t5)
    """

    def __init__(self, encoder: nn.Module, projection: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.projection = projection

    def forward(self, landmarks: torch.Tensor, landmarks_mask: torch.Tensor) -> torch.Tensor:
        enc_out = self.encoder(landmarks, landmarks_mask)
        return self.projection(enc_out)


def export_torchscript(
    model: ExportableLandmarkEncoder,
    dummy_landmarks: torch.Tensor,
    dummy_mask: torch.Tensor,
    output_path: Path,
) -> Path:
    """Exports model to TorchScript via tracing."""
    logger.info("Tracing model with TorchScript...")
    model.eval()
    with torch.no_grad():
        traced_model = torch.jit.trace(model, (dummy_landmarks, dummy_mask))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced_model.save(str(output_path))
    logger.info("Saved TorchScript model to %s (%.2f MB)", output_path, output_path.stat().st_size / (1024 * 1024))

    # Verify parity
    with torch.no_grad():
        eager_out = model(dummy_landmarks, dummy_mask)
        loaded = torch.jit.load(str(output_path))
        ts_out = loaded(dummy_landmarks, dummy_mask)
        diff = torch.max(torch.abs(eager_out - ts_out)).item()
        logger.info("TorchScript parity check: max absolute diff = %.6e", diff)
        assert diff < 1e-4, f"Parity check failed: max diff = {diff}"

    return output_path


def export_onnx(
    model: ExportableLandmarkEncoder,
    dummy_landmarks: torch.Tensor,
    dummy_mask: torch.Tensor,
    output_path: Path,
) -> Path | None:
    """Exports model to ONNX format with dynamic sequence lengths if supported by opset."""
    logger.info("Exporting model to ONNX...")
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_names = ["landmarks", "landmarks_mask"]
    output_names = ["projected_features"]
    dynamic_axes = {
        "landmarks": {0: "batch_size", 1: "sequence_length"},
        "landmarks_mask": {0: "batch_size", 1: "sequence_length"},
        "projected_features": {0: "batch_size", 1: "sequence_length"},
    }

    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                (dummy_landmarks, dummy_mask),
                str(output_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=14,
                do_constant_folding=True,
            )
        logger.info("Saved ONNX model to %s (%.2f MB)", output_path, output_path.stat().st_size / (1024 * 1024))
        return output_path
    except Exception as exc:
        logger.warning(
            "ONNX export not supported by current PyTorch fused transformer kernel (%s). "
            "Using TorchScript (.pt) as the primary deployment format.", exc
        )
        return None


def benchmark_export(
    pipeline: ASLInferencePipeline,
    torchscript_path: Path | None = None,
    num_runs: int = 20,
    seq_len: int = 120,
) -> dict[str, Any]:
    """Benchmarks end-to-end inference and encoder latency."""
    logger.info("Running export latency benchmark (seq_len=%d, runs=%d)...", seq_len, num_runs)
    stats = pipeline.benchmark_latency(num_runs=num_runs, seq_len=seq_len, beam_size=5)

    encoder_stats = {}
    if torchscript_path and torchscript_path.exists():
        device = pipeline.device
        ts_model = torch.jit.load(str(torchscript_path), map_location=device)
        ts_model.eval()
        dummy_lm = torch.randn(1, seq_len, 1200, device=device)
        dummy_mask = torch.ones(1, seq_len, dtype=torch.bool, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = ts_model(dummy_lm, dummy_mask)

        if device.type == "cuda":
            torch.cuda.synchronize()

        ts_times = []
        with torch.no_grad():
            for _ in range(num_runs):
                t0 = time.perf_counter()
                _ = ts_model(dummy_lm, dummy_mask)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                ts_times.append((time.perf_counter() - t0) * 1000.0)

        encoder_stats = {
            "mean_encoder_ms": round(float(np.mean(ts_times)), 2),
            "p95_encoder_ms": round(float(np.percentile(ts_times, 95)), 2),
            "min_encoder_ms": round(float(np.min(ts_times)), 2),
            "max_encoder_ms": round(float(np.max(ts_times)), 2),
        }
        logger.info("TorchScript Landmark Encoder standalone latency: Mean=%.2f ms", encoder_stats["mean_encoder_ms"])

    return {
        "end_to_end": stats,
        "encoder_standalone": encoder_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ASL Translation Models")
    parser.add_argument("--config", default="configs/t5_decoder.yaml", help="Path to model config")
    parser.add_argument("--checkpoint", default="checkpoints/t5/best_model.pt", help="Model checkpoint")
    parser.add_argument("--output-dir", default="checkpoints/export", help="Output export directory")
    parser.add_argument("--format", choices=["torchscript", "onnx", "all"], default="all", help="Export target format")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark after export")
    parser.add_argument("--seq-len", type=int, default=120, help="Benchmark sequence length in frames")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize inference pipeline
    pipeline = ASLInferencePipeline(
        model_type="t5",
        config_path=args.config,
        checkpoint_path=args.checkpoint,
    )

    exportable_encoder = ExportableLandmarkEncoder(
        encoder=pipeline.model.encoder,
        projection=pipeline.model.projection,
    ).to(pipeline.device)
    exportable_encoder.eval()

    dummy_lm = torch.randn(1, args.seq_len, 1200, device=pipeline.device)
    dummy_mask = torch.ones(1, args.seq_len, dtype=torch.bool, device=pipeline.device)

    ts_path = out_dir / "landmark_encoder.pt"
    onnx_path = out_dir / "landmark_encoder.onnx"

    if args.format in ("torchscript", "all"):
        export_torchscript(exportable_encoder, dummy_lm, dummy_mask, ts_path)

    if args.format in ("onnx", "all"):
        export_onnx(exportable_encoder, dummy_lm, dummy_mask, onnx_path)

    if args.benchmark:
        bench_results = benchmark_export(pipeline, ts_path, num_runs=15, seq_len=args.seq_len)
        print("\n" + "=" * 60)
        print("LATENCY BENCHMARK RESULTS")
        print("=" * 60)
        print(yaml.dump(bench_results, sort_keys=False))


if __name__ == "__main__":
    main()
