# Restart report: from sentence translation (T5) to sign recognition + Qwen

## Why the old system produced random text
1. Task/data mismatch: How2Sign = open-vocabulary sentences, ~16k extracted clips. The T5 model's validation loss was
   3.35 (real signs) vs 3.48 (all-zero input) vs 3.62 (another clip's signs): the encoder carried ~no information; the
   model was an unconditional language model (stock "So, you're going to want to..." outputs).
2. Not a code bug: same model memorises 64 clips to loss 0.004 (Gate-2 overfit test) but cannot generalise.
   A bigger decoder (Qwen) does not fix missing information; it only hallucinates more fluently.
3. Train/serve skew bugs (fixed):
   - normalize.py turned undetected (all-zero) hands into a fake hand at -shoulder_mid/scale; JS sent exact zeros.
   - hand/face visibility channels: Python wrote 0.0, JS wrote 1.0 (338 channels differ).
   - lost pose gave scale 1e-6 (values blow up).

## Current system
landmarks (1200-dim, 15 fps) -> sign classifier (ensemble of 3 small transformers, 2,304 words + `_none_` class)
-> multi-scale windows + NMS, restricted to the active word list -> word sequence
-> Qwen2.5-1.5B-Instruct + LoRA fine-tuned for gloss->English -> faithfulness guard -> English sentence.

## Phase 2: word recognition (more words, higher accuracy, no overfitting)
- Data: WLASL + ASL Citizen (83,399 webcam clips, 52 signers). All 83,388 usable clips extracted (11 unreadable).
  Words with >= 10 training clips: 2,304. Official signer-disjoint splits: test signers are never seen in training.
- Model choices made on VALIDATION loss only: 4 configs swept with early stopping; the most regularised one won
  (d_model 192, 2 layers, dropout 0.3, weight decay 0.1, mirror augmentation, hand-shape features).
  Augmented train loss 1.56 vs val loss 0.93 at the chosen epoch: no overfitting gap.
- Deployed: 3 seeds retrained on train+val for a fixed 100 epochs (= mean early-stopping epoch).

| test signers (never seen) | old (WLASL-141) | train-only ensemble | deployed train+val ensemble |
|---|---|---|---|
| 141 conversational words, WLASL signers | 66.7% | 86.7% | **88.9%** |
| same words, ASL Citizen signers | - | 92.5% | **93.8%** |
| 366-word demo list (default), ASL Citizen | - | 87.7% | **89.4%** (top-5 98.4%) |
| 366-word demo list, WLASL | - | 80.1% | **81.9%** (top-5 97.4%) |
| all 2,304 words | - | 74.2% | **76.1%** (top-5 94.6%) |

Chained streams of 3 unseen-signer clips: word recall 85%, exact sequence 56% (direct recognizer);
through the real WebSocket server: recall 84%, exact 70%. Decoder thresholds (min_conf 0.5, overlap 0.2) tuned on
validation streams, not test.

## Phase 3: sentences (gloss -> English)
- Held-out test set: 115 hand-written ASL-gloss/English pairs (data/sentences/eval_gloss_en.tsv), never trained on.
- Training data: 24k clean gloss/English pairs generated from an English grammar over the vocabulary
  (scripts/make_sentence_data.py): tense from time words, WH-word last, NOT before the verb, "YOU ..." yes/no
  questions, plurals, possessives. Pairs whose gloss is in the test set were removed.
- LoRA (r=16) on Qwen2.5-1.5B-Instruct, loss on the answer only, adapter with the lowest validation loss kept.

| composer (115 held-out sentences) | signed words kept | sentences with invented content | question form right | chrF |
|---|---|---|---|---|
| Qwen2.5-1.5B few-shot (stock) | 98.8% | 5.2% | 92.2% | 74.2 |
| Qwen2.5-3B few-shot (stock) | 98.3% | 4.3% | 97.4% | 81.0 |
| **Qwen2.5-1.5B + LoRA (deployed)** | **99.8%** | **0.0%** | **99.1%** | **87.7** |

Runtime guard: if a sentence drops a signed word or adds a content word, the recognised words are shown instead.

## How to run
`python -m src.serve.server --port 8000` (defaults: sign classifier, 366-word demo vocabulary, fine-tuned composer).
`--vocab all` enables all 2,304 words (76% instead of ~89% top-1). Tests: `pytest tests`, `scripts/e2e_test.py`,
`scripts/ws_test.py`, `scripts/eval_vocab.py checkpoints/sign/final`, `scripts/eval_composer.py --adapter checkpoints/composer/lora_1.5b`.

## Honest limits
- Accuracy is measured on recorded test clips; live webcam results depend on lighting, framing and signing speed.
- Yes/no questions in ASL are marked by raised eyebrows, not by a sign; the composer assumes "YOU ..." without a WH-word
  is a question.
- The sentence test set and the training grammar were both written by us, so the composer scores are likely
  optimistic for very free-form signing; random word combinations fall back to a word list.
- Pausing with hands down for over ~1.4 s ends a sentence; sign continuously within one sentence.
- Previous models are kept: checkpoints/sign/final_wlasl141, checkpoints/sign/final_cit_trainonly.
