"""
Shared Landmark Encoder for ASL sequences.
Can be used by LSTM baseline, Transformer baseline, and T5 translator.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LandmarkEncoder(nn.Module):
    """
    Encodes landmark frames (B, T, in_features=1200) into hidden representations (B, T, d_model).
    Supports either 'transformer' or 'lstm' backbone.
    """

    def __init__(
        self,
        in_features: int = 1200,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        backbone: str = "transformer",
    ):
        super().__init__()
        self.in_features = in_features
        self.d_model = d_model
        self.backbone = backbone

        # Initial linear projection from landmark feature dim to d_model
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        if backbone == "transformer":
            self.pos_encoder = PositionalEncoding(d_model, max_len=1024, dropout=dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
                enable_nested_tensor=False,
            )
        elif backbone == "lstm":
            self.encoder = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model // 2,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=True,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(
        self,
        landmarks: torch.Tensor,
        landmarks_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        landmarks: (B, T, in_features)
        landmarks_mask: (B, T) bool tensor (True for real frames, False for pad)
        Returns: (B, T, d_model)
        """
        x = self.input_proj(landmarks)  # (B, T, d_model)

        if self.backbone == "transformer":
            x = self.pos_encoder(x)
            src_key_padding_mask = None
            if landmarks_mask is not None:
                # TransformerEncoder expects True for positions that should be masked (ignored)
                src_key_padding_mask = ~landmarks_mask
            out = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        else:  # lstm
            out, _ = self.encoder(x)

        return out
