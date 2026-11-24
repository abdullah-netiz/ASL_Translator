"""Isolated-sign classifier on the shared 1200-dim landmark features."""
from __future__ import annotations
import torch, torch.nn as nn
from src.models.landmark_encoder import LandmarkEncoder


TIPS = [4, 8, 12, 16, 20]
_PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]


def hand_shape_features(x: torch.Tensor) -> torch.Tensor:
    """(B,T,1200) -> (B,T,146): per hand, wrist-relative shape normalised by hand size (63)
    + 10 fingertip distances. Computed in-model, so the 1200-dim wire format is unchanged."""
    feats = []
    for s in (472, 535):
        h = x[..., s:s + 63].reshape(*x.shape[:-1], 21, 3)
        size = h[..., 9, :].norm(dim=-1, keepdim=True).clamp_min(1e-3)          # wrist -> middle MCP
        present = (h.abs().sum((-1, -2)) > 0).float().unsqueeze(-1)
        hn = h / size.unsqueeze(-1)
        tips = hn[..., TIPS, :]
        d = torch.stack([(tips[..., i, :] - tips[..., j, :]).norm(dim=-1) for i, j in _PAIRS], -1)
        feats += [hn.flatten(-2) * present, d * present]
    return torch.cat(feats, -1)


class SignClassifier(nn.Module):
    def __init__(self, num_classes: int, in_features: int = 1200, d_model: int = 256,
                 nhead: int = 8, layers: int = 4, dropout: float = 0.2, feat: str = 'all', derived: bool = False):
        super().__init__()
        self.cfg = dict(num_classes=num_classes, in_features=in_features, d_model=d_model,
                        nhead=nhead, layers=layers, dropout=dropout, feat=feat, derived=derived)
        self.derived = derived
        self.encoder = LandmarkEncoder(in_features + (146 if derived else 0), d_model, nhead, layers, 4 * d_model, dropout, "transformer")
        keep = torch.ones(in_features)
        if feat in ("noface", "hands"):        # drop face block (static 236:472 + its velocity)
            keep[236:472] = 0; keep[836:1072] = 0
        if feat == "hands":                    # also drop pose head points (0-10); keep arms/shoulders
            keep[0:44] = 0; keep[600:644] = 0
        self.register_buffer("keep", keep)
        self.attn = nn.Linear(d_model, 1)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = x * self.keep
        if self.derived:
            z = torch.cat([z, hand_shape_features(x)], -1)
        h = self.encoder(z, mask)                        # (B,T,D)
        w = self.attn(h).squeeze(-1).masked_fill(~mask, -1e4).softmax(-1)
        return self.head((w.unsqueeze(-1) * h).sum(1))
