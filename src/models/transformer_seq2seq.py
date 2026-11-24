"""
Transformer Seq2Seq Baseline trained from scratch for ASL translation.
Includes beam search (width 5) and greedy decoding.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.landmark_encoder import PositionalEncoding


class TransformerSeq2Seq(nn.Module):
    """
    Standard Transformer Encoder-Decoder for sign language translation.
    """

    def __init__(
        self,
        in_features: int = 1200,
        vocab_size: int = 8000,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pad_idx: int = 0,
        bos_idx: int = 1,
        eos_idx: int = 2,
    ):
        super().__init__()
        self.in_features = in_features
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

        # Landmark projection & Encoder
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_encoder = PositionalEncoding(d_model, max_len=1024, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Token embedding & Decoder
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_decoder = PositionalEncoding(d_model, max_len=512, dropout=dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Output projection
        self.out_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
        self,
        landmarks: torch.Tensor,
        landmarks_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        landmarks: (B, T, in_features)
        landmarks_mask: (B, T) bool tensor
        Returns: memory (B, T, d_model)
        """
        x = self.input_proj(landmarks)
        x = self.pos_encoder(x)
        src_key_padding_mask = ~landmarks_mask if landmarks_mask is not None else None
        memory = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return memory

    def decode(
        self,
        targets: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        targets: (B, L) token IDs
        memory: (B, T, d_model)
        Returns: logits (B, L, vocab_size)
        """
        x = self.embedding(targets) * math.sqrt(self.d_model)
        x = self.pos_decoder(x)
        out = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.out_proj(out)

    def forward(
        self,
        landmarks: torch.Tensor,
        landmarks_mask: torch.Tensor | None,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        landmarks: (B, T, in_features)
        targets: (B, L) full token sequence (BOS .. EOS)
        Returns: logits (B, L-1, vocab_size) predicting targets[:, 1:]
        """
        device = targets.device
        dec_input = targets[:, :-1]
        L_in = dec_input.size(1)

        # Causal mask for decoder self-attention
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(L_in, device=device).bool()
        tgt_key_padding_mask = (dec_input == self.pad_idx)
        memory_key_padding_mask = ~landmarks_mask if landmarks_mask is not None else None

        memory = self.encode(landmarks, landmarks_mask)
        logits = self.decode(
            targets=dec_input,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return logits

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
        Generate translations using beam search (width 5 by default).
        If beam_size == 1, performs fast greedy search.
        """
        self.eval()
        if beam_size <= 1:
            return self._greedy_generate(landmarks, landmarks_mask, max_len=max_len)
        return self._beam_search(landmarks, landmarks_mask, beam_size=beam_size, max_len=max_len, length_penalty=length_penalty)

    @torch.no_grad()
    def _greedy_generate(
        self,
        landmarks: torch.Tensor,
        landmarks_mask: torch.Tensor | None,
        max_len: int,
    ) -> list[list[int]]:
        B = landmarks.size(0)
        device = landmarks.device
        memory = self.encode(landmarks, landmarks_mask)
        memory_key_padding_mask = ~landmarks_mask if landmarks_mask is not None else None

        generated = torch.full((B, 1), fill_value=self.bos_idx, dtype=torch.long, device=device)
        finished = [False] * B

        for _ in range(max_len):
            L = generated.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(L, device=device).bool()
            logits = self.decode(
                targets=generated,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            for b in range(B):
                if next_token[b].item() == self.eos_idx:
                    finished[b] = True
            if all(finished):
                break

        results = []
        for b in range(B):
            tokens = []
            for tid in generated[b, 1:].tolist():
                if tid == self.eos_idx:
                    break
                tokens.append(tid)
            results.append(tokens)
        return results

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
        memory = self.encode(landmarks, landmarks_mask)
        memory_key_padding_mask = ~landmarks_mask if landmarks_mask is not None else None

        results: list[list[int]] = []

        for b in range(B):
            mem_b = memory[b:b+1]  # (1, T, d_model)
            mem_mask_b = memory_key_padding_mask[b:b+1] if memory_key_padding_mask is not None else None

            # Each beam candidate: (log_prob, [token_ids])
            beams: list[tuple[float, list[int]]] = [(0.0, [self.bos_idx])]
            completed: list[tuple[float, list[int]]] = []

            for _ in range(max_len):
                candidates: list[tuple[float, list[int]]] = []
                for score, tokens in beams:
                    if tokens[-1] == self.eos_idx:
                        completed.append((score, tokens))
                        continue

                    tgt = torch.tensor([tokens], dtype=torch.long, device=device)
                    L = len(tokens)
                    tgt_mask = nn.Transformer.generate_square_subsequent_mask(L, device=device).bool()
                    logits = self.decode(
                        targets=tgt,
                        memory=mem_b,
                        tgt_mask=tgt_mask,
                        memory_key_padding_mask=mem_mask_b,
                    )
                    log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
                    topk_probs, topk_ids = torch.topk(log_probs, k=beam_size)

                    for p, idx in zip(topk_probs.tolist(), topk_ids.tolist()):
                        candidates.append((score + p, tokens + [idx]))

                if not candidates:
                    break

                # Sort candidates by length-penalized score
                def beam_score(item: tuple[float, list[int]]) -> float:
                    raw_score, seq = item
                    lp = ((5.0 + len(seq)) / 6.0) ** length_penalty
                    return raw_score / lp

                candidates.sort(key=beam_score, reverse=True)
                beams = candidates[:beam_size]

                # Stop if top beam is already completed
                if len(completed) >= beam_size:
                    break

            completed.extend(beams)
            completed.sort(key=lambda x: x[0] / (((5.0 + len(x[1])) / 6.0) ** length_penalty), reverse=True)
            best_seq = completed[0][1]
            # Strip BOS and EOS
            cleaned_tokens = [t for t in best_seq[1:] if t != self.eos_idx]
            results.append(cleaned_tokens)

        return results
