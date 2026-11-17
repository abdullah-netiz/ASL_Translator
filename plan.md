# plan.md — Real-Time ASL-to-English Translation

Work through phases in order. Do not start a phase until the previous gate is passed. Rules live in `instructions.md`.

## Phase 0 — Setup

- Create repo layout, environment, pinned `requirements.txt`, config system, logging.
- Download Green Screen RGB clips (frontal): train, val, test, plus the English translation CSVs.
- Verify: every CSV row maps to a video file; list and count any mismatches.

**Gate 0:** clip counts and CSV row counts reconcile per split. Missing files are listed.

## Phase 1 — Landmark extraction

- Implement MediaPipe Holistic extractor with the keypoint subset, normalization, 15 fps sampling.
- Run on all splits in parallel (multi-process, GPU or CPU as available). Cache `.npy` per clip.
- Run QC: hand detection rate, sequence length distribution, failures.
- Raw 31G of video shrinks to a few GB of landmarks. After this step, training never touches video again.

**Gate 1:** under ~5% of clips are failed or mostly empty; length distribution reviewed and max length chosen.

## Phase 2 — Data pipeline and tokenizer

- Dataset class, collate with padding and masks, length bucketing.
- Text cleaning (lowercase optional, normalize whitespace). Train tokenizer on train text only.
- Implement augmentations.
- Overfit sanity test: train on ~50 clips until loss approaches zero.

**Gate 2:** the model can memorize 50 clips. If not, there is a bug. Fix before scaling.

## Phase 3 — Baselines

- Train LSTM encoder-decoder with attention.
- Train Transformer encoder-decoder from scratch.
- Evaluate on val: BLEU, ROUGE-L, sample outputs.

**Gate 3:** both baselines train stably and produce non-trivial output. Record val BLEU as the reference floor.

## Phase 4 — Main model on the 31G data

- Transformer landmark encoder + pretrained T5 decoder (T5-small, then T5-base).
- Tune on val: learning rate, depth, dropout, augmentation strength, max length, fps, landmark subset (ablations).
- Run final evaluation on test once and write `reports/phase4.md`.

**Gate 4 (decides the 290G question):**
- If val BLEU keeps improving with more training data (check a learning curve using 25%, 50%, 100% of train) -> scaling data is justified, go to Phase 5.
- If val BLEU plateaus or train/val gap is large -> more data of the same kind may not help. Improve the model, augmentation, or pretraining instead, and revisit.

## Phase 5 — Scale up (only if Gate 4 allows)

- Re-check what the larger downloads add: full-length videos, more views, or additional modalities. Confirm they pair with the same translation text and do not overlap with val/test.
- Extend extraction and retrain the best config. Compare against Phase 4 on the same val and test sets.

**Gate 5:** measurable val improvement over Phase 4. If none, stay with the smaller setup.

## Phase 6 — Real-time system

- Export the best model (TorchScript or ONNX), benchmark latency.
- FastAPI + WebSocket inference server.
- Browser client: webcam capture, MediaPipe JS landmarks, same normalization as Python. Run the parity test.
- Sentence boundary detection for continuous signing.
- Dashboard: live preview, current sentence, transcript, confidence.

**Gate 6:** end-to-end demo works on a live webcam; latency measured and reported; parity test passes.

## Phase 7 — Evaluation, report, polish

- Final metrics table: LSTM vs Transformer vs main model.
- Failure analysis: long sentences, fingerspelling, names and numbers, poor lighting, non-studio backgrounds.
- Document limitations (studio data vs real webcam), demo video, README with setup steps.

## Deliverables checklist

- [ ] Reproducible extraction and training scripts with configs
- [ ] Cached landmarks plus QC report
- [ ] Three trained models with val and test metrics
- [ ] Learning-curve evidence for the 290G decision
- [ ] Working real-time web dashboard
- [ ] Final report with honest limitations
