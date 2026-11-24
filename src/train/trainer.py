"""
Trainer class for ASL translation models.
Supports mixed precision, gradient clipping, label smoothing, warmup+cosine LR, and validation eval.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.eval.metrics import compute_metrics, format_sample_table
from src.train.schedulers import get_cosine_schedule_with_warmup

logger = logging.getLogger(__name__)


class Trainer:
    """
    Standard training & evaluation loop.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        tokenizer: Any,
        checkpoint_dir: str | Path,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-2,
        warmup_steps: int = 1000,
        max_epochs: int = 50,
        grad_clip: float = 1.0,
        label_smoothing: float = 0.1,
        mixed_precision: bool = True,
        device: str | torch.device | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_epochs = max_epochs
        self.grad_clip = grad_clip
        self.mixed_precision = mixed_precision and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        total_steps = len(train_loader) * max_epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=min(warmup_steps, max(1, total_steps // 10)),
            num_training_steps=max(1, total_steps),
        )

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=label_smoothing,
        )

        self.best_val_loss = float("inf")
        self.best_bleu = 0.0

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        total_tokens = 0

        for batch in self.train_loader:
            landmarks = batch["landmarks"].to(self.device)
            lm_mask = batch["landmarks_mask"].to(self.device)
            tokens = batch["tokens"].to(self.device)
            target_labels = batch["target_labels"].to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                if hasattr(self.model, "t5"):
                    outputs = self.model(landmarks, landmarks_mask=lm_mask, labels=target_labels)
                    loss = outputs.loss
                    non_pad = (target_labels != -100).sum().item()
                else:
                    logits = self.model(landmarks, lm_mask, tokens)
                    targets = target_labels[:, 1:].contiguous()
                    loss = self.criterion(logits.view(-1, self.tokenizer.vocab_size), targets.view(-1))
                    non_pad = (targets != -100).sum().item()

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item() * non_pad
            total_tokens += non_pad

        return total_loss / max(1, total_tokens)

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader | None = None,
        beam_size: int = 5,
        max_samples: int = 100,
    ) -> dict[str, Any]:
        eval_loader = loader or self.val_loader
        if eval_loader is None or len(eval_loader) == 0:
            return {"loss": 0.0, "metrics": {}}

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        clip_ids: list[str] = []
        references: list[str] = []
        hypotheses: list[str] = []

        for batch in eval_loader:
            landmarks = batch["landmarks"].to(self.device)
            lm_mask = batch["landmarks_mask"].to(self.device)
            tokens = batch["tokens"].to(self.device)
            target_labels = batch["target_labels"].to(self.device)

            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                if hasattr(self.model, "t5"):
                    outputs = self.model(landmarks, landmarks_mask=lm_mask, labels=target_labels)
                    loss = outputs.loss
                    non_pad = (target_labels != -100).sum().item()
                else:
                    logits = self.model(landmarks, lm_mask, tokens)
                    targets = target_labels[:, 1:].contiguous()
                    loss = self.criterion(logits.view(-1, self.tokenizer.vocab_size), targets.view(-1))
                    non_pad = (targets != -100).sum().item()

            total_loss += loss.item() * non_pad
            total_tokens += non_pad

            # Generate translations
            if len(hypotheses) < max_samples:
                preds = self.model.generate(landmarks, lm_mask, beam_size=beam_size)
                for pred_ids, ref_text, cid in zip(preds, batch["texts"], batch["clip_ids"]):
                    hyp_text = self.tokenizer.decode(pred_ids, skip_special_tokens=True)
                    hypotheses.append(hyp_text)
                    references.append(ref_text)
                    clip_ids.append(cid)

        val_loss = total_loss / max(1, total_tokens)
        metrics = compute_metrics(hypotheses, references)

        return {
            "loss": val_loss,
            "metrics": metrics,
            "clip_ids": clip_ids,
            "hypotheses": hypotheses,
            "references": references,
        }

    def fit(self, eval_every: int = 1) -> dict[str, Any]:
        logger.info(f"Starting training on {self.device} for {self.max_epochs} epochs...")
        history: list[dict] = []

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_info = {}

            if self.val_loader and (epoch % eval_every == 0 or epoch == self.max_epochs):
                val_info = self.evaluate(self.val_loader, beam_size=1)  # fast eval during training
                val_loss = val_info["loss"]
                bleu = val_info["metrics"].get("bleu", 0.0)

                logger.info(
                    f"Epoch {epoch:2d}/{self.max_epochs:2d} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val BLEU: {bleu:.2f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_bleu = bleu
                    self.save_checkpoint("best_model.pt", epoch=epoch, val_loss=val_loss, bleu=bleu)
            else:
                logger.info(
                    f"Epoch {epoch:2d}/{self.max_epochs:2d} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_info.get("loss"),
                "val_bleu": val_info.get("metrics", {}).get("bleu"),
            })

        self.save_checkpoint("last_model.pt", epoch=self.max_epochs)
        return {"history": history, "best_val_loss": self.best_val_loss, "best_bleu": self.best_bleu}

    def save_checkpoint(self, filename: str, **extra: Any) -> Path:
        path = self.checkpoint_dir / filename
        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "extra": extra,
        }
        torch.save(state, path)
        return path
