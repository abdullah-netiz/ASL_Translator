"""
LSTM Seq2Seq Baseline with Bahdanau Attention.
Maps landmark sequence (B, T, 1200) -> English token sequence (B, L).
"""
from __future__ import annotations

import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """Additive attention: score(s, h) = v^T tanh(W_s * s + W_h * h)"""

    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int):
        super().__init__()
        self.w_enc = nn.Linear(enc_dim, attn_dim, bias=False)
        self.w_dec = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        query : (B, dec_dim)
        keys  : (B, T, enc_dim)
        mask  : (B, T) bool (True for valid keys)
        Returns:
            context : (B, enc_dim)
            weights : (B, T)
        """
        # (B, 1, attn_dim) + (B, T, attn_dim) -> (B, T, attn_dim)
        scores = self.v(torch.tanh(self.w_dec(query).unsqueeze(1) + self.w_enc(keys))).squeeze(2)

        if mask is not None:
            fill_val = -1e4 if scores.dtype in (torch.float16, torch.bfloat16) else -1e9
            scores = scores.masked_fill(~mask, fill_val)

        weights = F.softmax(scores, dim=-1)  # (B, T)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)  # (B, enc_dim)
        return context, weights


class LSTMSeq2Seq(nn.Module):
    """
    Encoder-Decoder LSTM for ASL translation.
    """

    def __init__(
        self,
        in_features: int = 1200,
        vocab_size: int = 8000,
        d_model: int = 256,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.2,
        pad_idx: int = 0,
        bos_idx: int = 1,
        eos_idx: int = 2,
    ):
        super().__init__()
        self.in_features = in_features
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

        # Landmark projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Bi-LSTM Encoder
        self.encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size // 2,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

        # Attention
        self.attention = BahdanauAttention(
            enc_dim=hidden_size,
            dec_dim=hidden_size,
            attn_dim=hidden_size // 2,
        )

        # Decoder
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.dec_cell = nn.LSTMCell(d_model + hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size + hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def encode(
        self,
        landmarks: torch.Tensor,
        landmarks_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = self.input_proj(landmarks)
        enc_outputs, (h_n, c_n) = self.encoder(x)  # enc_outputs: (B, T, hidden_size)

        # Combine bidirectional states for decoder initial state
        # h_n is (num_layers * 2, B, hidden_size // 2)
        h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (B, hidden_size)
        c_last = torch.cat([c_n[-2], c_n[-1]], dim=-1)  # (B, hidden_size)
        return enc_outputs, (h_last, c_last)

    def forward(
        self,
        landmarks: torch.Tensor,
        landmarks_mask: torch.Tensor | None,
        targets: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        landmarks      : (B, T, in_features)
        landmarks_mask : (B, T) bool
        targets        : (B, L) token IDs (starts with BOS)
        Returns: logits of shape (B, L-1, vocab_size)
        """
        B, T, _ = landmarks.shape
        _, L = targets.shape

        enc_outputs, (h, c) = self.encode(landmarks, landmarks_mask)
        context = torch.zeros(B, self.hidden_size, device=landmarks.device)

        logits_list: list[torch.Tensor] = []
        current_token = targets[:, 0]  # BOS

        for t in range(1, L):
            emb = self.embedding(current_token)  # (B, d_model)
            cell_in = torch.cat([emb, context], dim=-1)
            h, c = self.dec_cell(cell_in, (h, c))

            context, _ = self.attention(h, enc_outputs, landmarks_mask)
            logit = self.out_proj(torch.cat([h, context], dim=-1))  # (B, vocab_size)
            logits_list.append(logit)

            # Teacher forcing decision
            if random.random() < teacher_forcing_ratio:
                current_token = targets[:, t]
            else:
                current_token = logit.argmax(dim=-1)

        return torch.stack(logits_list, dim=1)  # (B, L-1, vocab_size)

    @torch.no_grad()
    def generate(
        self,
        landmarks: torch.Tensor,
        landmarks_mask: torch.Tensor | None = None,
        beam_size: int = 5,
        max_len: int = 64,
        length_penalty: float = 0.6,
    ) -> list[list[int]]:
        """
        Generate translations using beam search or greedy decoding.
        """
        self.eval()
        if beam_size <= 1:
            return self._greedy_generate(landmarks, landmarks_mask, max_len=max_len)
        return self._beam_search(landmarks, landmarks_mask, beam_size=beam_size, max_len=max_len, length_penalty=length_penalty)

    @torch.no_grad()
    def _greedy_generate(
        self,
        landmarks: torch.Tensor,
        landmarks_mask: torch.Tensor | None = None,
        max_len: int = 64,
    ) -> list[list[int]]:
        """Greedy autoregressive decoding."""
        B = landmarks.size(0)
        enc_outputs, (h, c) = self.encode(landmarks, landmarks_mask)
        context = torch.zeros(B, self.hidden_size, device=landmarks.device)

        current_token = torch.full((B,), fill_value=self.bos_idx, dtype=torch.long, device=landmarks.device)
        predictions = [[] for _ in range(B)]
        finished = [False] * B

        for _ in range(max_len):
            emb = self.embedding(current_token)
            cell_in = torch.cat([emb, context], dim=-1)
            h, c = self.dec_cell(cell_in, (h, c))
            context, _ = self.attention(h, enc_outputs, landmarks_mask)
            logit = self.out_proj(torch.cat([h, context], dim=-1))
            next_token = logit.argmax(dim=-1)

            for b in range(B):
                if not finished[b]:
                    tid = next_token[b].item()
                    if tid == self.eos_idx:
                        finished[b] = True
                    else:
                        predictions[b].append(tid)

            if all(finished):
                break
            current_token = next_token

        return predictions

    @torch.no_grad()
    def _beam_search(
        self,
        landmarks: torch.Tensor,
        landmarks_mask: torch.Tensor | None,
        beam_size: int = 5,
        max_len: int = 64,
        length_penalty: float = 0.6,
    ) -> list[list[int]]:
        B = landmarks.size(0)
        device = landmarks.device
        enc_outputs, (h_init, c_init) = self.encode(landmarks, landmarks_mask)

        results: list[list[int]] = []
        for b in range(B):
            enc_b = enc_outputs[b:b+1]
            mask_b = landmarks_mask[b:b+1] if landmarks_mask is not None else None
            h_b = h_init[b:b+1]
            c_b = c_init[b:b+1]
            ctx_b = torch.zeros(1, self.hidden_size, device=device)

            # Each candidate: (score, [token_ids], h, c, ctx)
            beams = [(0.0, [self.bos_idx], h_b, c_b, ctx_b)]
            completed: list[tuple[float, list[int]]] = []

            for _ in range(max_len):
                candidates = []
                for score, tokens, h, c, ctx in beams:
                    if tokens[-1] == self.eos_idx:
                        completed.append((score, tokens))
                        continue

                    curr = torch.tensor([tokens[-1]], dtype=torch.long, device=device)
                    emb = self.embedding(curr)
                    cell_in = torch.cat([emb, ctx], dim=-1)
                    new_h, new_c = self.dec_cell(cell_in, (h, c))
                    new_ctx, _ = self.attention(new_h, enc_b, mask_b)
                    logit = self.out_proj(torch.cat([new_h, new_ctx], dim=-1))
                    log_probs = F.log_softmax(logit[0], dim=-1)
                    topk_probs, topk_ids = torch.topk(log_probs, k=beam_size)

                    for p, idx in zip(topk_probs.tolist(), topk_ids.tolist()):
                        candidates.append((score + p, tokens + [idx], new_h, new_c, new_ctx))

                if not candidates:
                    break

                def lp_score(item):
                    s, seq = item[0], item[1]
                    lp = ((5.0 + len(seq)) / 6.0) ** length_penalty
                    return s / lp

                candidates.sort(key=lp_score, reverse=True)
                beams = candidates[:beam_size]
                if len(completed) >= beam_size:
                    break

            completed.extend([(b[0], b[1]) for b in beams])
            completed.sort(key=lambda x: x[0] / (((5.0 + len(x[1])) / 6.0) ** length_penalty), reverse=True)
            best_seq = completed[0][1]
            results.append([t for t in best_seq[1:] if t != self.eos_idx])

        return results
