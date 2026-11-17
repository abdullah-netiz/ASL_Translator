"""
src/serve/inference.py — Real-Time Inference Engine for ASL Translation.

Wraps trained sequence models (T5ASLTranslator, TransformerSeq2Seq, LSTMSeq2Seq)
for low-latency single-sequence translation from landmark arrays to English text.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

from src.datasets.tokenizer import ASLTokenizer
from src.models.lstm_seq2seq import LSTMSeq2Seq
from src.models.t5_translator import T5ASLTranslator
from src.models.transformer_seq2seq import TransformerSeq2Seq

logger = logging.getLogger("inference")


class ASLInferencePipeline:
    """
    End-to-end inference wrapper for ASL-to-English translation.
    Takes normalized landmark frame sequences (T, 1200) and produces English sentences.
    """

    def __init__(
        self,
        model_type: str = "t5",
        config_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        beam_size: int = 5,
        max_length: int = 128,
    ):
        self.model_type = model_type.lower()
        self.beam_size = beam_size
        self.max_length = max_length

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info("Initializing ASLInferencePipeline [%s] on %s", self.model_type, self.device)

        # Resolve paths
        repo_root = Path(__file__).resolve().parents[2]
        if config_path is None:
            if self.model_type == "t5":
                config_path = repo_root / "configs" / "t5_decoder.yaml"
            elif self.model_type == "transformer":
                config_path = repo_root / "configs" / "transformer.yaml"
            elif self.model_type == "lstm":
                config_path = repo_root / "configs" / "lstm.yaml"
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")

        if checkpoint_path is None:
            checkpoint_path = repo_root / "checkpoints" / self.model_type / "best_model.pt"

        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)

        with open(self.config_path) as f:
            self.cfg = yaml.safe_load(f)

        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self) -> None:
        """Loads model weights and tokenizer based on model_type."""
        repo_root = Path(__file__).resolve().parents[2]

        if self.model_type == "t5":
            t5_name = self.cfg.get("t5_model", "t5-small")
            logger.info("Loading T5 tokenizer: %s", t5_name)
            self.tokenizer = AutoTokenizer.from_pretrained(t5_name)

            self.model = T5ASLTranslator(
                t5_model_name=t5_name,
                in_features=self.cfg.get("in_features", 1200),
                encoder_d_model=self.cfg.get("encoder_d_model", 256),
                encoder_nhead=self.cfg.get("encoder_nhead", 8),
                encoder_layers=self.cfg.get("encoder_layers", 4),
                encoder_dropout=self.cfg.get("encoder_dropout", 0.1),
            )
        elif self.model_type == "transformer":
            tok_model_path = repo_root / "data" / "tokenizer" / "spm_bpe_8000.model"
            self.tokenizer = ASLTokenizer(tok_model_path)
            self.model = TransformerSeq2Seq(
                in_features=self.cfg.get("in_features", 1200),
                vocab_size=self.cfg.get("vocab_size", 8000),
                d_model=self.cfg.get("d_model", 256),
                nhead=self.cfg.get("nhead", 8),
                num_encoder_layers=self.cfg.get("num_encoder_layers", 4),
                num_decoder_layers=self.cfg.get("num_decoder_layers", 4),
                dim_feedforward=self.cfg.get("dim_feedforward", 1024),
                dropout=self.cfg.get("dropout", 0.1),
            )
        elif self.model_type == "lstm":
            tok_model_path = repo_root / "data" / "tokenizer" / "spm_bpe_8000.model"
            self.tokenizer = ASLTokenizer(tok_model_path)
            self.model = LSTMSeq2Seq(
                in_features=self.cfg.get("in_features", 1200),
                vocab_size=self.cfg.get("vocab_size", 8000),
                hidden_dim=self.cfg.get("hidden_dim", 256),
                embed_dim=self.cfg.get("embed_dim", 128),
                num_layers=self.cfg.get("num_layers", 2),
                dropout=self.cfg.get("dropout", 0.2),
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Load weights
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        logger.info("Loading model weights from %s", self.checkpoint_path)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully and set to eval mode")

    @torch.no_grad()
    def translate(
        self,
        landmarks: np.ndarray | torch.Tensor,
        beam_size: int | None = None,
        max_length: int | None = None,
    ) -> dict[str, Any]:
        """
        Translates a single landmark sequence into English text.

        Args:
            landmarks: Array/Tensor of shape (T, 1200) or (1, T, 1200)
            beam_size: Beam search width (defaults to pipeline config)
            max_length: Maximum tokens to generate

        Returns:
            Dict containing translated text, token count, latency_ms, etc.
        """
        beam = beam_size if beam_size is not None else self.beam_size
        max_len = max_length if max_length is not None else self.max_length

        if isinstance(landmarks, np.ndarray):
            lm_tensor = torch.from_numpy(landmarks).float()
        else:
            lm_tensor = landmarks.float()

        if lm_tensor.ndim == 2:
            lm_tensor = lm_tensor.unsqueeze(0)  # (1, T, 1200)
        elif lm_tensor.ndim != 3 or lm_tensor.size(0) != 1:
            raise ValueError(f"Expected input shape (T, 1200) or (1, T, 1200), got {lm_tensor.shape}")

        if lm_tensor.size(-1) != 1200:
            raise ValueError(f"Expected 1200 landmark features, got {lm_tensor.size(-1)}")

        T = lm_tensor.size(1)
        if T == 0:
            return {
                "text": "",
                "tokens": [],
                "num_frames": 0,
                "latency_ms": 0.0,
                "confidence": 0.0,
            }

        # Truncate if excessively long
        max_frames = self.cfg.get("max_frames", 300)
        if T > max_frames:
            lm_tensor = lm_tensor[:, :max_frames, :]
            T = max_frames

        lm_tensor = lm_tensor.to(self.device)
        mask = torch.ones((1, T), dtype=torch.bool, device=self.device)

        # Synchronize GPU for accurate latency measurement
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        if self.model_type == "t5":
            generated = self.model.generate(
                landmarks=lm_tensor,
                landmarks_mask=mask,
                beam_size=beam,
                max_length=max_len,
            )
            token_ids = generated[0]
            if hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        else:
            # Baseline models
            generated = self.model.generate(
                src=lm_tensor,
                src_mask=mask,
                max_len=max_len,
                beam_size=beam,
            )
            token_ids = generated[0]
            text = self.tokenizer.decode(token_ids)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        return {
            "text": text,
            "tokens": token_ids,
            "num_tokens": len(token_ids),
            "num_frames": T,
            "latency_ms": round(latency_ms, 2),
            "fps_processed": round(T / (latency_ms / 1000.0), 1) if latency_ms > 0 else 0,
        }

    def benchmark_latency(
        self,
        num_runs: int = 20,
        seq_len: int = 120,
        beam_size: int = 5,
    ) -> dict[str, float]:
        """
        Runs latency benchmark on synthetic sequences to measure real-time compliance.
        Target: < 2000 ms end-to-end on GPU.
        """
        logger.info(
            "Benchmarking %s inference: %d runs, seq_len=%d, beam=%d...",
            self.model_type, num_runs, seq_len, beam_size
        )
        dummy_input = np.random.randn(seq_len, 1200).astype(np.float32)

        # Warmup (3 passes)
        for _ in range(3):
            self.translate(dummy_input, beam_size=beam_size)

        latencies = []
        for _ in range(num_runs):
            res = self.translate(dummy_input, beam_size=beam_size)
            latencies.append(res["latency_ms"])

        latencies_arr = np.array(latencies)
        stats = {
            "mean_ms": round(float(np.mean(latencies_arr)), 2),
            "std_ms": round(float(np.std(latencies_arr)), 2),
            "median_ms": round(float(np.median(latencies_arr)), 2),
            "p95_ms": round(float(np.percentile(latencies_arr, 95)), 2),
            "min_ms": round(float(np.min(latencies_arr)), 2),
            "max_ms": round(float(np.max(latencies_arr)), 2),
            "target_ms": 2000.0,
            "passed_target": bool(np.mean(latencies_arr) < 2000.0),
        }
        logger.info(
            "Benchmark Result: Mean=%.2f ms, P95=%.2f ms (Target: < 2000 ms -> %s)",
            stats["mean_ms"], stats["p95_ms"], "PASSED" if stats["passed_target"] else "FAILED"
        )
        return stats
