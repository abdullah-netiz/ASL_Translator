"""
Tokenizer for ASL-to-English translation.

Uses SentencePiece BPE trained strictly on the TRAIN split.
Provides text cleaning, tokenization, batch encoding/decoding,
and special tokens (<pad>=0, <bos>=1, <eos>=2, <unk>=3).
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import sentencepiece as spm

# ---------------------------------------------------------------------------
# Text normalization & cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str, lowercase: bool = False) -> str:
    """
    Normalize text:
      - Strip leading/trailing whitespace
      - Collapse multiple whitespace to single space
      - Normalize quotes and dashes
      - Optional lowercasing
    """
    if not text:
        return ""
    # Replace unicode quotes and dashes with ASCII
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = text.replace("—", "-").replace("–", "-")
    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text).strip()
    if lowercase:
        text = text.lower()
    return text


# ---------------------------------------------------------------------------
# SentencePiece Wrapper
# ---------------------------------------------------------------------------

class ASLTokenizer:
    """
    SentencePiece tokenizer wrapper with special token IDs.
    """

    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3

    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self, model_path: str | Path | None = None, lowercase: bool = False):
        self.lowercase = lowercase
        self.sp = spm.SentencePieceProcessor()
        self.model_path = str(model_path) if model_path else None
        if self.model_path and Path(self.model_path).exists():
            self.sp.load(self.model_path)

    @property
    def is_loaded(self) -> bool:
        return len(self.sp) > 0

    @property
    def vocab_size(self) -> int:
        return len(self.sp) if self.is_loaded else 0

    @property
    def pad_token_id(self) -> int:
        return self.PAD_ID

    @property
    def bos_token_id(self) -> int:
        return self.BOS_ID

    @property
    def eos_token_id(self) -> int:
        return self.EOS_ID

    @property
    def unk_token_id(self) -> int:
        return self.UNK_ID

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        """Encode string to token IDs with optional BOS/EOS."""
        if not self.is_loaded:
            raise RuntimeError("Tokenizer model is not loaded.")
        cleaned = clean_text(text, lowercase=self.lowercase)
        ids = self.sp.encode(cleaned, out_type=int)
        tokens = []
        if add_bos:
            tokens.append(self.BOS_ID)
        tokens.extend(ids)
        if add_eos:
            tokens.append(self.EOS_ID)
        return tokens

    def decode(
        self,
        ids: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs back to string."""
        if not self.is_loaded:
            raise RuntimeError("Tokenizer model is not loaded.")
        special_ids = {self.PAD_ID, self.BOS_ID, self.EOS_ID}
        filtered: list[int] = []
        for tid in ids:
            if skip_special_tokens and tid in special_ids:
                continue
            filtered.append(int(tid))
        return self.sp.decode(filtered)

    @classmethod
    def train(
        cls,
        sentences: list[str],
        output_prefix: str | Path,
        vocab_size: int = 8000,
        model_type: str = "bpe",
        character_coverage: float = 0.9995,
        lowercase: bool = False,
    ) -> "ASLTokenizer":
        """
        Train a SentencePiece model on a list of sentences (train split only).
        """
        cleaned_sentences = [clean_text(s, lowercase=lowercase) for s in sentences if s.strip()]
        if not cleaned_sentences:
            raise ValueError("No valid sentences provided for tokenizer training.")

        # SentencePiece requires vocab_size <= num_unique_words or reasonable bound
        # Ensure vocab size does not exceed reasonable threshold for small datasets
        actual_vocab_size = min(vocab_size, max(50, len(set(" ".join(cleaned_sentences).split())) * 2))
        actual_vocab_size = min(vocab_size, actual_vocab_size)

        output_prefix = str(output_prefix)
        Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
            for s in cleaned_sentences:
                f.write(s + "\n")
            temp_corpus = f.name

        try:
            spm.SentencePieceTrainer.train(
                input=temp_corpus,
                model_prefix=output_prefix,
                vocab_size=actual_vocab_size,
                model_type=model_type,
                character_coverage=character_coverage,
                pad_id=cls.PAD_ID,
                bos_id=cls.BOS_ID,
                eos_id=cls.EOS_ID,
                unk_id=cls.UNK_ID,
                pad_piece=cls.PAD_TOKEN,
                bos_piece=cls.BOS_TOKEN,
                eos_piece=cls.EOS_TOKEN,
                unk_piece=cls.UNK_TOKEN,
                user_defined_symbols=[],
            )
        finally:
            if os.path.exists(temp_corpus):
                os.remove(temp_corpus)

        model_file = f"{output_prefix}.model"
        return cls(model_path=model_file, lowercase=lowercase)


# ---------------------------------------------------------------------------
# Training CLI
# ---------------------------------------------------------------------------

def train_tokenizer_from_manifest(
    train_manifest_path: str | Path,
    output_dir: str | Path,
    vocab_size: int = 8000,
    model_type: str = "bpe",
    lowercase: bool = False,
) -> Path:
    """Read train manifest sentences only and train SentencePiece model."""
    train_manifest_path = Path(train_manifest_path)
    if not train_manifest_path.exists():
        raise FileNotFoundError(f"Train manifest not found: {train_manifest_path}")

    sentences: list[str] = []
    with open(train_manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = row.get("sentence", "").strip()
            if s:
                sentences.append(s)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f"spm_{model_type}_{vocab_size}"

    tokenizer = ASLTokenizer.train(
        sentences=sentences,
        output_prefix=prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        lowercase=lowercase,
    )
    model_path = Path(f"{prefix}.model")
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer on train manifest")
    parser.add_argument("--manifest", default="data/manifests/train_manifest.csv",
                        help="Path to train manifest CSV (MUST be train split only)")
    parser.add_argument("--out-dir", default="data/tokenizer",
                        help="Output directory for tokenizer model")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--model-type", default="bpe", choices=["bpe", "unigram"])
    parser.add_argument("--lowercase", action="store_true")
    args = parser.parse_args()

    model_path = train_tokenizer_from_manifest(
        train_manifest_path=args.manifest,
        output_dir=args.out_dir,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        lowercase=args.lowercase,
    )
    print(f"Tokenizer trained successfully: {model_path}")

    # Smoke test encoding/decoding
    tok = ASLTokenizer(model_path)
    sample = "Hello world! This is continuous sign language translation."
    encoded = tok.encode(sample)
    decoded = tok.decode(encoded)
    print(f"Sample:  {sample}")
    print(f"Encoded: {encoded[:10]}... (len {len(encoded)})")
    print(f"Decoded: {decoded}")


if __name__ == "__main__":
    main()
