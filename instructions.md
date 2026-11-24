# instructions.md — Real-Time ASL-to-English Translation

Read this file fully before doing anything. Follow `plan.md` for the order of work. If the two ever conflict, this file wins on rules and `plan.md` wins on sequencing.

## 1. Project goal

Build a webcam-based application that captures continuous American Sign Language (ASL) and shows the English translation on a web dashboard.

- Dataset: How2Sign (https://how2sign.github.io)
- Pipeline: video -> MediaPipe Holistic landmarks -> sequence model -> English sentence
- Task type: sign language TRANSLATION (ASL video -> free-form English), not gloss recognition and not isolated-sign classification.

## 2. Data rules (non-negotiable)

1. Use the **Green Screen RGB clips (frontal view)** download first. These are sentence-level clips, so each clip pairs with one English sentence.
   - Train (~31G), Validation (~1.7G), Test (~2.2G)
2. Also download the matching **English translation CSV files** (train/val/test) from the How2Sign site. The video files alone are useless without them. Match rows to clips using `SENTENCE_NAME`.
3. Split discipline:
   - Train split: fit model weights only.
   - Validation split: hyperparameter tuning, early stopping, checkpoint selection.
   - Test split: touched ONCE, at the end of each major phase, for reporting. Never tune on it.
4. Do not train on the test split, ever. Do not leak val/test sentences into the tokenizer vocabulary.
5. Do not move to the ~290G options until the Phase 4 gate in `plan.md` is met. Log the decision and the reason.
6. Store raw data under `data/raw/`, extracted landmarks under `data/landmarks/`, never commit either to git.

## 3. Landmark extraction rules

- Use MediaPipe Holistic. Pin the exact mediapipe version in `requirements.txt` and record it in `configs/`. The Holistic API has been moving to the Tasks API, so version drift will silently change outputs.
- Extract once, cache to disk as `.npy` (float16 is fine), one file per clip. Never re-run extraction inside the training loop.
- Keep these points per frame:
  - Pose: upper body only (nose, eyes, ears, shoulders, elbows, wrists, hips optional)
  - Left hand: 21 points
  - Right hand: 21 points
  - Face: a reduced subset only (lips, eyebrows, eyes, face contour, roughly 40-80 points). Do NOT keep all 468.
- Features per point: x, y, z plus a visibility/presence flag. Missing hand = zeros plus a `mask` flag, not interpolated garbage.
- Normalization (required):
  - Translate so the mid-shoulder point is the origin.
  - Scale by shoulder width.
  - Normalize hands also relative to their own wrist for an extra hand-shape stream.
- Add velocity (frame t minus frame t-1) as extra features.
- Sample at ~15 fps (stride 2 on 30 fps video). Cap sequences at a configurable max length (start at 256 frames) and log how many clips exceed it.
- Write an extraction QC script: per-clip fraction of frames with detected hands, and flag clips under a threshold. Report the stats before training.

## 4. Modeling rules

- Build baselines in this order, each with a reproducible config:
  1. LSTM encoder-decoder with attention (the required baseline)
  2. Transformer encoder-decoder trained from scratch
  3. **Main model:** Transformer landmark encoder + pretrained text decoder (T5-small first, T5-base if GPU time allows), connected through a learned projection
- Text side: use the pretrained model's own tokenizer for option 3; SentencePiece (~8k vocab, trained on TRAIN text only) for options 1 and 2.
- Training defaults: AdamW, warmup + cosine schedule, label smoothing 0.1, mixed precision (bf16), gradient clipping 1.0, dropout 0.1-0.3.
- Augmentation (landmark space): small rotation, scale, translation jitter, Gaussian noise, random frame dropout, mild speed change (0.9x-1.1x). Do NOT horizontally mirror (it swaps handedness and breaks signs).
- Decoding: beam search (width 5) for evaluation. Greedy is only for quick debugging.
- Metrics: sacreBLEU (BLEU-1..4), ROUGE-L, and a sample table of predictions vs references. Report on val during training and on test at gates.
- Track everything with a logger (Weights & Biases or TensorBoard). Seeds fixed and recorded.

## 5. Real-time application rules

- Architecture: browser captures webcam -> landmarks -> WebSocket -> FastAPI inference server -> translated text pushed back to the dashboard.
- Prefer running MediaPipe in the browser (JS) and sending only landmark arrays to the server, to cut bandwidth and latency. The JS landmark pipeline MUST apply the exact same selection and normalization as the Python training pipeline. Write a parity test that compares both on the same video.
- Continuous signing: use a sliding window plus a boundary detector (hands dropping or low-motion pause) to cut sentences. Do not wait for a button press.
- Export the model (TorchScript or ONNX) and measure end-to-end latency. Target under ~2 seconds from end of sentence to displayed text on a single GPU.
- Dashboard shows: live webcam preview, current translated sentence, running transcript, and a confidence indicator.

## 6. Engineering rules

- Python 3.10+, PyTorch. Project layout:
  ```
  configs/  data/  src/{extract,datasets,models,train,eval,serve}/  app/  notebooks/  tests/  reports/
  ```
- Every script is runnable from the CLI with a config file. No hardcoded paths.
- Write unit tests for: landmark normalization, dataset loading, tokenizer, and the JS/Python parity check.
- Commit small, descriptive changes. Keep `reports/` updated after each phase (metrics, curves, sample outputs, known failures).

## 7. Honesty and reporting rules

- Expect modest scores. Landmark-only models on How2Sign typically land in the low single digits to roughly 10 BLEU-4. Do not tune on the test set to look better, and do not claim numbers that were not measured.
- If a phase gate fails, stop, diagnose, and write down what you found. Do not silently skip to the next phase.
- When unsure about a design decision with real cost (switching datasets, changing the split, changing the landmark set), stop and ask the user instead of guessing.
- Report limitations plainly: How2Sign is studio green-screen, frontal, instructional-domain signing. Webcam performance on real-world backgrounds and other domains will be worse, and the demo should say so.
