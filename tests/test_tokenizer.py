"""
Unit tests for tokenizer (src/datasets/tokenizer.py).

Verifies:
  1. Text cleaning (whitespace normalization, quote normalization)
  2. Special token IDs: <pad>=0, <bos>=1, <eos>=2, <unk>=3
  3. Encoding produces BOS at index 0 and EOS at index -1
  4. Decoding correctly round-trips text (skipping special tokens)
  5. Training on dummy sentences succeeds and preserves special IDs
"""
from pathlib import Path
import tempfile
import pytest

from src.datasets.tokenizer import clean_text, ASLTokenizer


def test_clean_text():
    raw = '  “Hello,   world!”  —   This is a   test.  ’   '
    cleaned = clean_text(raw)
    assert cleaned == '"Hello, world!" - This is a test. \''


def test_special_token_constants():
    assert ASLTokenizer.PAD_ID == 0
    assert ASLTokenizer.BOS_ID == 1
    assert ASLTokenizer.EOS_ID == 2
    assert ASLTokenizer.UNK_ID == 3


def test_tokenizer_train_and_encode_decode():
    corpus = [
        "Hello world, this is sign language translation.",
        "American sign language continuous signing recognition.",
        "Deep learning model for webcam translation.",
        "SentencePiece tokenization trained on English text only.",
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = Path(tmpdir) / "test_spm"
        tok = ASLTokenizer.train(
            sentences=corpus,
            output_prefix=prefix,
            vocab_size=60,
            model_type="bpe",
        )

        assert tok.is_loaded
        assert tok.vocab_size > 0
        assert tok.pad_token_id == 0
        assert tok.bos_token_id == 1
        assert tok.eos_token_id == 2
        assert tok.unk_token_id == 3

        text = "Hello world"
        encoded = tok.encode(text, add_bos=True, add_eos=True)
        assert encoded[0] == 1  # BOS
        assert encoded[-1] == 2  # EOS

        decoded = tok.decode(encoded, skip_special_tokens=True)
        assert "Hello" in decoded
        assert "world" in decoded


def test_prebuilt_model_loading_if_exists():
    model_path = Path("data/tokenizer/spm_bpe_8000.model")
    if model_path.exists():
        tok = ASLTokenizer(model_path)
        assert tok.is_loaded
        assert tok.vocab_size == 8000
        encoded = tok.encode("Thank you very much.", add_bos=True, add_eos=True)
        assert encoded[0] == 1
        assert encoded[-1] == 2
        decoded = tok.decode(encoded)
        assert decoded == "Thank you very much."
