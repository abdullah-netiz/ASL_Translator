# How this project works, and what to learn to explain it

This is a study guide for presenting the project. For every part of the system it says **what we did**, **which
concept that is**, and **exactly what to search on YouTube** to learn it. Videos marked ★ are the best starting
points. Search the title as written; the channel name is in brackets.

Contents:
1. The project in one minute
2. How we built it, step by step
3. Concepts to learn, in pipeline order (with search terms)
4. Algorithms used, in one line each
5. Code walkthrough: which file does what
6. Questions teachers are likely to ask, with short answers
7. Suggested learning order (about 2–3 days)

---

## 1. The project in one minute

A webcam app that understands American Sign Language (ASL) and writes English.

```
camera → MediaPipe finds body/hand points (in the browser) → 1,200 numbers per frame
       → our transformer classifier recognises each SIGN (2,304 words) → list of words ("ME GO SCHOOL TOMORROW")
       → Qwen language model, fine-tuned by us, writes a sentence → "I will go to school tomorrow."
```

There are two kinds of AI in it:

| Part | Type of AI | Built by |
|---|---|---|
| Sign classifier | Small transformer, **trained from scratch by us** (1.8M parameters) | Us |
| Sentence writer | Large language model Qwen2.5 (1.5B parameters), **fine-tuned by us with LoRA** | Alibaba (base) + us (fine-tune) |

Results on people the model never saw: **89%** of signs correct on the 366-word demo list (98% in the top 5).
**0%** invented words in sentences.

---

## 2. How we built it, step by step

### Step 0: the first version failed, and we found out why
The first version tried to translate whole sentences directly (How2Sign videos → T5 model → English). It output
random text. We tested it scientifically:
- We fed it **real** signing, **all-zero** input, and **someone else's** signing. The loss was almost the same
  (3.35 / 3.48 / 3.62), so the model was ignoring the video and just guessing common English.
- It could memorise 64 clips perfectly (an "overfit test"), so the code worked. The *task* was too hard for the data:
  open-vocabulary sentences with only ~16k clips.

**Concepts:** ablation study, overfitting test, sign language *translation* vs *recognition*.

### Step 1: we fixed bugs where training data and live data differed
The browser (JavaScript) and the training code (Python) computed the features slightly differently:
- missing hands became fake hands;
- one value was 0 vs 1;
- a lost body pose divided by almost zero.

Any mismatch means the model sees different numbers live than in training.

**Concept:** train/serve skew, feature normalisation, unit tests (`tests/test_js_python_parity.py`).

### Step 2: changed the task to word recognition (WLASL dataset)
Instead of whole sentences, we recognise **individual signs** (isolated sign recognition), then let a language model
build the sentence. We used WLASL, a dataset of single-sign videos, with official splits where *test signers are
different people*. That gave 66–71% accuracy on 141 words.

### Step 3: made it work on continuous signing
Real users sign several words in a row. We:
- slide windows of 3 sizes over the video and classify each window (**sliding window**);
- added a **"_none_" class** ("no sign here") trained on resting hands and transitions;
- keep the most confident, non-overlapping windows (**non-maximum suppression**);
- cut sentences when the hands rest for ~1.4 s (**segmenter / state machine**).

Getting all 3 signs exactly right in a chained 3-sign test went from 10% to 37%.

### Step 4: more data and better training (ASL Citizen dataset)
- Added ASL Citizen: 83,399 webcam videos from 52 people.
- Extracted landmarks from all of them (about 9.5 h).
- Compared 4 model sizes and kept the one with the lowest **validation loss**. The smallest, most regularised one
  won.
- Trained 3 copies with different random seeds and averaged them (**ensemble**).
- Retrained on train + validation data for deployment.
- Added **mirror augmentation** (left-handed signers) and **hand-shape features**.

Accuracy on the original 141 words went from 66.7% to 88.9%, and the vocabulary grew to 2,304 words.

### Step 5: taught the language model to write sentences (fine-tuning)
- Wrote 115 test sentences by hand (never used for training).
- Generated 24,000 training pairs (gloss → English) from a grammar we coded.
- Fine-tuned Qwen2.5-1.5B with **LoRA**.

It went from 5.2% sentences with invented words to **0%**, and beat the bigger Qwen 3B. A **safety check** shows
only the recognised words if a sentence would add anything that wasn't signed.

### Step 6: fixes after user testing
- **"It keeps saying DOG":** limiting to the demo word list re-scaled probabilities, so unknown signs were forced
  onto the closest listed word. We added a raw-probability threshold, which cut wrong words for unknown signs from
  47% to ~19%.
- **Word / Sentence mode switch:** Word mode shows one word per sign; Sentence mode builds a sentence.

### Step 7: deployment
- FastAPI web server with WebSockets.
- Shared over HTTPS with a Cloudflare tunnel, so phones can use their camera.
- Documentation, and a cloud notebook for testing bigger models.

---

## 3. Concepts to learn, in pipeline order

### 3.1 Machine-learning basics (learn first)

| Concept | Where in our project | Search on YouTube |
|---|---|---|
| What a neural network is | Everything | ★ "But what is a neural network?" (3Blue1Brown) |
| Gradient descent | Training | ★ "Gradient descent, how neural networks learn" (3Blue1Brown) |
| Backpropagation | Training | "3Blue1Brown backpropagation" (Deep learning chapter 3) |
| Supervised learning, classification | Sign classifier | "Supervised vs unsupervised learning" |
| Train / validation / test split | All results | "Train test validation split explained" |
| Softmax and argmax | Classifier output | ★ "Neural Networks Part 5: ArgMax and SoftMax" (StatQuest) |
| Cross-entropy loss | Classifier training | ★ "Neural Networks Part 6: Cross Entropy" (StatQuest) |
| Overfitting vs underfitting | Why we chose a small model | "Overfitting and underfitting explained" |
| Early stopping | Keep the lowest validation loss | "Early stopping neural network" |
| Dropout | `--dropout 0.3` | "Dropout regularization deep learning" (Andrew Ng, C2W1L06/L07) |
| Weight decay (L2 regularisation) | `--wd 0.1` | "Regularization Part 1: Ridge (L2) Regression" (StatQuest) |
| Label smoothing | Training loss | "Label smoothing explained" |
| Adam / AdamW optimiser | Training | "Adam optimizer explained" (Andrew Ng: "Adam Optimization Algorithm C2W2L08") |
| Learning-rate schedule (warmup + cosine) | Training | "learning rate warmup cosine decay" |
| Mini-batches, epochs | Training | "Mini-batch gradient descent" (Andrew Ng, C2W2L01) |
| Data augmentation | Rotation, scale, speed, mirror | "Data augmentation explained" |
| Ensemble learning (averaging models) | 3-model ensemble | "Ensemble learning explained" |
| Hyper-parameter search | 4-config sweep | "Hyperparameter tuning explained" |
| Data leakage / signer-independent split | Test signers never seen | "Data leakage in machine learning" |
| Ablation study | Diagnosing the first version | "What is an ablation study machine learning" |

### 3.2 Computer vision: from camera to numbers

| Concept | Where | Search |
|---|---|---|
| Pose estimation / keypoints | MediaPipe landmarks | "Human pose estimation explained" |
| MediaPipe Holistic (body + hands + face) | Browser + extraction | ★ "Sign Language Detection using ACTION RECOGNITION with Python LSTM Deep Learning Model" (Nicholas Renotte). This is the closest tutorial to our project |
| Landmark normalisation (centre + scale) | `normalize.py`, `landmarks.js` | "Normalizing pose keypoints translation scale invariance" |
| Velocity features (frame differences) | Features 600–1199 | "Motion features frame difference" |
| Frame-rate resampling | 15 fps | "Video frame rate resampling" |
| Train/serve skew | Step 1 bug fixes | "Training serving skew machine learning" |

### 3.3 The sign classifier (sequence model)

| Concept | Where | Search |
|---|---|---|
| Sequence data, time series classification | A sign = sequence of frames | "Time series classification deep learning" |
| Transformer architecture | `SignClassifier` | ★ "Transformer Neural Networks, ChatGPT's foundation, Clearly Explained!!!" (StatQuest) |
| Self-attention | Inside the transformer | ★ "3Blue1Brown attention in transformers" (Deep learning chapter 6) |
| Multi-head attention | 6 heads | "Multi-head attention explained" |
| Positional encoding | Frame order | "Positional encoding transformer explained" |
| Padding and attention masks | Clips have different lengths | "attention mask padding transformer" |
| Attention pooling | Turning frames into one vector | "attention pooling sequence classification" |
| Layer normalisation, residual connections | Inside the transformer | "Layer normalization explained" |
| Mixed precision (fp16 / bf16) | GPU training | "Mixed precision training explained" |

### 3.4 Continuous recognition (several signs in a row)

| Concept | Where | Search |
|---|---|---|
| Sliding window | `recognizer._windows` | "Sliding window algorithm" / "sliding window object detection" |
| Non-maximum suppression (NMS) | `recognizer.recognize` | ★ "Non-max Suppression (C4W3L07)" (Andrew Ng) |
| Background / rejection class | `_none_` class | "open set recognition" / "background class object detection" |
| Confidence threshold, precision vs recall | `min_conf`, `min_raw` | ★ "Precision and recall" / "ROC and AUC, Clearly Explained!" (StatQuest) |
| Out-of-distribution detection | The "dog" fix | "Out of distribution detection explained" |
| Temporal segmentation | Sentence cutting | "temporal action segmentation" |
| Finite state machine | Segmenter: idle ↔ signing | "Finite state machine explained" |

### 3.5 Language model: words → sentence

| Concept | Where | Search |
|---|---|---|
| Large language models | Qwen2.5 | ★ "Large Language Models explained briefly" (3Blue1Brown) |
| How GPT-style models generate text | Composer | ★ "Let's build GPT: from scratch, in code, spelled out." (Andrej Karpathy). Long; watch the first part |
| Tokenisation | Text → tokens | "Let's build the GPT Tokenizer" (Andrej Karpathy) |
| Instruction-tuned / chat models, system prompt | `SYSTEM_FT`, chat template | "instruction tuning LLM explained" |
| Few-shot prompting | Stock model prompt (`SHOTS`) | "few shot prompting explained" |
| Greedy decoding vs beam search | `do_sample=False` | "Greedy decoding vs beam search" |
| Fine-tuning | `train_composer.py` | "Fine-tuning LLMs explained" |
| **LoRA (low-rank adaptation)** | Our sentence model | ★ "LoRA explained" / "LoRA low-rank adaptation of large language models explained" |
| Synthetic training data | `make_sentence_data.py` | "synthetic data generation for NLP" |
| Hallucination and guardrails | `faithful.py` safety check | "LLM hallucination explained" |
| Lemmatisation | Faithfulness check | "Stemming vs lemmatization NLP" |
| BLEU / chrF metrics | Sentence evaluation | "BLEU score explained" (chrF is the character-level version) |

### 3.6 ASL linguistics (teachers may ask)

| Concept | Search |
|---|---|
| ASL gloss (writing signs as English words in capitals) | "ASL gloss explained" |
| ASL sentence order (time first, topic–comment, WH-question at the end) | "ASL sentence structure" |
| Non-manual markers (raised eyebrows = yes/no question) | "ASL non-manual markers" |
| Isolated sign recognition vs continuous sign language translation | "isolated sign language recognition vs continuous" |

### 3.7 Software and deployment

| Concept | Where | Search |
|---|---|---|
| Client–server architecture, REST API | FastAPI server | "REST API explained" |
| FastAPI | `src/serve/server.py` | "FastAPI tutorial for beginners" |
| WebSockets (live two-way streaming) | `/ws/translate` | ★ "WebSockets in 100 Seconds" (Fireship) |
| Binary data / Float32Array | 4,800-byte frames | "JavaScript typed arrays Float32Array" |
| async / await, threads | Server keeps streaming while recognising | "Python async await explained" |
| HTTPS and camera permission (getUserMedia) | Why a tunnel is needed | "getUserMedia https secure context" |
| Tunnelling (Cloudflare Tunnel) | Sharing to phones | "Cloudflare Tunnel explained" |
| GPU, CUDA, VRAM | Training / shared PC | "CUDA explained" / "Why GPUs for deep learning" |
| Unit testing with pytest | `tests/` | "pytest tutorial" |
| Git and GitHub | Version control | "Git and GitHub for beginners" (freeCodeCamp) |

---

## 4. Algorithms used, in one line each

| Algorithm | One-line description | File |
|---|---|---|
| Landmark normalisation | Move origin to the mid-shoulder point, divide by shoulder width, express hands relative to the wrist | `src/extract/normalize.py` |
| Velocity features | Feature at frame t minus feature at frame t−1 | `src/sign/data.py: with_velocity` |
| Hand-shape features | Finger points scaled by palm size + 10 fingertip distances | `src/sign/model.py: hand_shape_features` |
| Transformer encoder + attention pooling | Frames → contextual vectors → weighted average → class scores | `src/sign/model.py: SignClassifier` |
| Cross-entropy with label smoothing | Training loss; targets 0.9/0.1 instead of 1/0 to avoid over-confidence | `src/sign/train.py` |
| AdamW + cosine schedule + early stopping | Optimiser, learning-rate decay, keep the lowest validation loss | `src/sign/train.py` |
| Augmentation (rotate, scale, shift, speed, crop, hand-drop, mirror) | Random changes so the model generalises | `src/sign/data.py` |
| Synthetic negatives | Resting / transition clips labelled `_none_` | `src/sign/data.py: make_negative` |
| Ensemble averaging | Mean of 3 models' softmax probabilities | `src/sign/recognizer.py: _windows` |
| Multi-scale sliding window | Windows of 20/30/42 frames every 3 frames | `src/sign/recognizer.py: _windows` |
| Greedy NMS | Keep the most confident window, drop overlapping ones, repeat | `src/sign/recognizer.py: recognize` |
| Active-vocabulary masking + raw-probability floor | Only listed words; reject signs whose real probability is low | `src/sign/recognizer.py` |
| Pause/motion segmentation | State machine: signing → 21 idle frames → cut | `src/serve/segmenter.py` |
| LoRA fine-tuning | Train small low-rank matrices added to Qwen's layers; base weights frozen | `scripts/train_composer.py` |
| Greedy decoding | Always pick the most likely next token | `src/sign/composer.py: compose` |
| Faithfulness check | Lemmatise; every signed word present and no new content word, else fall back | `src/sign/faithful.py` |
| Grammar-based data generation | Templates + conjugation rules → 24k gloss/English pairs | `scripts/make_sentence_data.py` |

---

## 5. Code walkthrough: which file does what

Follow the data from the camera to the screen:

| Step | File | Key functions / what to say |
|---|---|---|
| 1. Camera + landmarks | `app/static/js/ui.js` | Starts the camera, runs MediaPipe Holistic, draws the skeleton, handles the Word/Sentence switch |
| 2. Features in browser | `app/static/js/landmarks.js` | Turns MediaPipe points into the 1,200-number vector (same maths as `normalize.py`) |
| 3. Streaming | `app/static/js/socket.js` | Sends each frame as 4,800 bytes over a WebSocket; receives `live_words` and `translation` |
| 4. Server | `src/serve/server.py` | `websocket_translate()`: receives frames, runs the segmenter, calls the recogniser, sends results |
| 5. Sentence boundary | `src/serve/segmenter.py` | `is_frame_active()` (hands visible + moving), `process_frame()` (state machine, cut after a pause) |
| 6. Recognition | `src/sign/recognizer.py` | `_windows()` (sliding windows + ensemble), `recognize()` (thresholds + NMS), `recognize_word()` (Word mode), `translate()` |
| 7. Model | `src/sign/model.py` | `SignClassifier`: feature selection → transformer → attention pooling → 2,305 class scores |
| 8. Sentence | `src/sign/composer.py` | `GlossComposer.compose()`: Qwen + LoRA writes the sentence; picks a model that fits the GPU |
| 9. Safety | `src/sign/faithful.py` | `score()`: checks the sentence keeps all signed words and adds none |
| Training data | `src/sign/data.py` | `load_manifest()` (splits), `augment()`, `mirror()`, `make_negative()`, `SignDataset` |
| Training | `src/sign/train.py` | Training loop, early stopping, saves `best.pt` + `results.json` |
| Sentence training | `scripts/make_sentence_data.py`, `scripts/train_composer.py` | Grammar generator; LoRA fine-tuning |
| Dataset extraction | `scripts/wlasl_extract.py`, `scripts/aslcitizen_extract.py` | Video → MediaPipe → normalised `.npy` files |
| Evaluation | `scripts/eval_vocab.py`, `e2e_test.py`, `ws_test.py`, `eval_composer.py` | Accuracy on unseen signers; sentence quality |
| Tests | `tests/` | `pytest`: normalisation, JS/Python parity, server |

Old first-version code, kept for reference: `src/models`, `src/train`, `src/eval`, `src/datasets`.

---

## 6. Questions teachers are likely to ask, with short answers

**Why not translate whole sentences directly like the first version?**
With ~16k sentence clips the model learned nothing from the video. We proved it: zero input gave almost the same
loss. Recognising words first and then using a language model needs far less data and is checkable.

**Why landmarks instead of raw video pixels?**
They are small (1,200 numbers per frame), private (no video leaves the browser), fast, and independent of
background, clothing and lighting.

**Why a transformer?**
A sign is a sequence of frames. Self-attention compares every frame with every other, which captures movement
patterns, and transformers train in parallel on the GPU. With attention pooling the model stays small
(1.8M parameters), which matters with only ~46k training clips. An LSTM would also be a valid choice; we did not
compare the two on the word task.

**How do you know it isn't overfitting?**
- Test signers are different people from the training signers.
- We chose models by validation loss and used early stopping.
- The smallest, most regularised model won.
- Training loss (1.56, with augmentation) stayed above validation loss (0.93).

**What is the `_none_` class for?**
Most windows in continuous signing contain half a sign or a transition. Without a "nothing" option the model is
forced to pick a word.

**What is NMS?**
Many overlapping windows detect the same sign. We keep the most confident one and remove overlapping ones. This is
the same idea as in object detection (YOLO).

**What is LoRA and why use it?**
Instead of retraining all 1.5 billion parameters, we train small extra matrices (18.5M parameters, 1.2%). It is
cheap, needs little GPU memory, and the base model keeps its general English skill.

**Why is your 1.5B better than the 3B?**
Ours is fine-tuned for exactly this task: 0% invented words vs 4.3% for the stock 3B, and chrF 87.7 vs 81.0 on 115
held-out sentences.

**Where did the training sentences come from?**
A grammar program generates 24k gloss/English pairs. The 115 test sentences were hand-written separately and
excluded from training.

**What happens with a sign the system doesn't know?**
The raw-probability floor rejects it (no output) instead of forcing the nearest known word.

**How does it know a sentence ended?**
After about 1.4 s with no hand motion or hands lowered (0.6 s in Word mode).

**What are the limitations?**
- No fingerspelling.
- Facial grammar is only approximated.
- Lighting and framing matter.
- Accuracy is measured on recorded clips, so live results vary.

**How accurate is it?**
On people it never saw:
- 89.4% top-1 and 98.4% top-5 on the 366 demo words;
- 76.1% on all 2,304 words;
- 84% of words found in 3-sign sequences through the live server.

---

## 7. Suggested learning order (about 2–3 days)

1. 3Blue1Brown neural network chapters 1–3, plus StatQuest softmax and cross-entropy (about 2 h).
2. Overfitting, dropout, early stopping, train/val/test (about 1 h).
3. Nicholas Renotte's sign language action-recognition video (about 1 h). It is closest to our pipeline.
4. StatQuest transformer + 3Blue1Brown attention (about 1.5 h).
5. Andrew Ng's non-max suppression video, plus precision vs recall (30 min).
6. 3Blue1Brown "LLMs explained briefly" + a LoRA explainer (1 h).
7. Fireship WebSockets + a FastAPI intro (30 min).
8. Read this file's section 5 with the code open, then [ARCHITECTURE.md](ARCHITECTURE.md) and
   [reports/restart_report.md](../reports/restart_report.md) for the numbers.
