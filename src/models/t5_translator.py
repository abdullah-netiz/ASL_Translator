"""
Phase 4 Main Model: Transformer Landmark Encoder + Pretrained T5 Decoder.

Architecture:
  - Input: Landmark sequences (B, T, 1200)
  - Landmark Encoder: Spatial projection + Positional Encoding + Transformer Encoder
  - Learned Projection: Projects encoder representations (B, T, d_enc) to T5 embedding space (B, T, d_t5)
  - Pretrained T5 Decoder: Generates English text autoregressively with cross-attention
"""
from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from src.models.landmark_encoder import LandmarkEncoder


class T5ASLTranslator(nn.Module):
    """
    Sign language translation model connecting a Transformer Landmark Encoder
    to a pretrained T5 text decoder through a learned projection layer.
    """

    def __init__(
        self,
        t5_model_name: str = "t5-small",
        in_features: int = 1200,
        encoder_d_model: int = 256,
        encoder_nhead: int = 8,
        encoder_layers: int = 4,
        encoder_dropout: float = 0.1,
        freeze_t5_decoder: bool = False,
    ):
        super().__init__()
        self.t5_model_name = t5_model_name

        # 1. Landmark Encoder
        self.encoder = LandmarkEncoder(
            in_features=in_features,
            d_model=encoder_d_model,
            nhead=encoder_nhead,
            num_layers=encoder_layers,
            dropout=encoder_dropout,
            backbone="transformer",
        )

        # 2. Pretrained T5 Model
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        self.d_t5 = self.t5.config.d_model  # 512 for t5-small, 768 for t5-base

        # 3. Learned Projection Layer (d_enc -> d_t5)
        self.projection = nn.Sequential(
            nn.Linear(encoder_d_model, self.d_t5),
            nn.LayerNorm(self.d_t5),
            nn.Dropout(encoder_dropout),
            nn.ReLU(),
            nn.Linear(self.d_t5, self.d_t5),
            nn.LayerNorm(self.d_t5),
        )

        if freeze_t5_decoder:
            for p in self.t5.decoder.parameters():
                p.requires_grad = False
            for p in self.t5.lm_head.parameters():
                p.requires_grad = False

    def encode_landmarks(
        self,
        landmarks: torch.Tensor,
        landmarks_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encodes (B, T, in_features) -> (B, T, d_t5) in T5 representation space.
        """
        enc_out = self.encoder(landmarks, landmarks_mask)  # (B, T, d_enc)
        projected = self.projection(enc_out)               # (B, T, d_t5)
        return projected

    def forward(
        self,
        landmarks: torch.Tensor,
        landmarks_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
    ) -> Any:
        """
        Training forward pass.
        landmarks      : (B, T, in_features)
        landmarks_mask : (B, T) bool
        labels         : (B, L) target token IDs with -100 for padding
        """
        projected = self.encode_landmarks(landmarks, landmarks_mask)
        encoder_outputs = BaseModelOutput(last_hidden_state=projected)

        # T5 expects 1 for valid tokens and 0 for padding tokens
        attention_mask = landmarks_mask.long() if landmarks_mask is not None else None

        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
        )
        return outputs

    @torch.no_grad()
    def generate(
        self,
        landmarks: torch.Tensor,
        landmarks_mask: torch.Tensor | None = None,
        beam_size: int = 5,
        max_length: int = 64,
        min_new_tokens: int = 3,
        length_penalty: float = 1.0,
        repetition_penalty: float = 1.3,
        no_repeat_ngram_size: int = 3,
    ) -> list[list[int]]:
        """
        Generate token IDs using T5's beam search (width 5 by default).
        """
        self.eval()
        projected = self.encode_landmarks(landmarks, landmarks_mask)
        encoder_outputs = BaseModelOutput(last_hidden_state=projected)
        attention_mask = landmarks_mask.long() if landmarks_mask is not None else None

        gen_kwargs: dict[str, Any] = {
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "num_beams": beam_size,
            "max_length": max_length,
            "min_new_tokens": min_new_tokens,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
        }
        if beam_size > 1:
            gen_kwargs["length_penalty"] = length_penalty
            gen_kwargs["early_stopping"] = True

        generated_ids = self.t5.generate(**gen_kwargs)

        return generated_ids.tolist()
