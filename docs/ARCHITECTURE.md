# Architecture

```
 Browser                                   Server (FastAPI, src/serve/server.py)
 ───────                                   ──────────────────────────────────────────────────────────────────────
 camera ─► MediaPipe Holistic (JS)         /ws/translate
        ─► landmarks.js normalisation ───► ContinuousSignSegmenter ── sentence ──► SignTranslator.translate()
           1200 float32 / frame, 15 fps      (pause / wrist-drop cut)               │
                                                                                     ├─ sliding windows (20/30/42 frames)
 dashboard ◄── translation / live_words ◄────────────────────────────────────────── ├─ 3-model SignClassifier ensemble
                                                                                     ├─ active-word mask + NMS ─► words
                                                                                     └─ GlossComposer (Qwen2.5-1.5B + LoRA)
                                                                                          └─ faithfulness check ─► English
```

Each stage is described below in pipeline order.

## 1. Landmarks (browser and Python, identical)

MediaPipe Holistic runs in the browser. `app/static/js/landmarks.js` and its Python twin `src/extract/normalize.py`
turn each frame into **1,200 numbers**. Raw video never reaches the server. `tests/test_js_python_parity.py`
checks that both implementations produce the same values.

| Index | Content |
|---|---|
| 0–67 | 17 upper-body pose points × (x, y, z, visibility) |
| 68–151 / 152–235 | left / right hand, 21 points × 4 |
| 236–471 | 59 face points × 4 |
| 472–534 / 535–597 | left / right hand shape relative to the wrist, 21 × 3 |
| 598–599 | left / right hand detected flags |
| 600–1199 | frame-to-frame velocity of 0–599 |

Normalisation: origin at the mid-shoulder point, scaled by shoulder width. When the pose is lost, the last good
centre and scale are reused. An undetected hand is exactly zero. Frames are resampled to 15 fps.

## 2. Sign classifier (`src/sign/model.py`)

`SignClassifier` labels one clip (one sign) with one of 2,304 words or `_none_` ("no sign / transition").

- **Input features (`feat="hands"`):** face and head-pose values are dropped. This removes signer identity cues
  and helps on new signers. 146 hand-shape features are added: per hand, wrist-relative shape scaled by palm
  size, plus 10 fingertip distances.
- **Network:** linear projection → transformer encoder (d_model 192, 2 layers, 6 heads; 1.8M parameters) → attention pooling →
  LayerNorm/Dropout/Linear head.
- **Deployed:** an ensemble of 3 such models (different seeds), probabilities averaged.
  Files: `checkpoints/sign/final/m0.pt … m2.pt`. Each file stores the weights, the config and the word list.

### Why a `_none_` class

During continuous signing many windows contain half a sign or a transition. Without a "nothing" option the
classifier forces one of the words, which produces phantom words. `_none_` is trained on synthetic negatives
(resting hands, transitions stitched from two clips). It raised exact 3-sign recovery from 10% to 37% when it was
added, before the larger model.

## 3. From a sentence to words (`src/sign/recognizer.py`)

`SignTranslator.recognize(segment)`:

1. Slide windows of 20, 30 and 42 frames (≈1.3–2.8 s) with a stride of 3 frames.
2. Classify every window with the ensemble.
3. Keep only the **active words**. These are the 366-word demo list (`data/aslcitizen/vocab_demo.txt`) unless the
   server is started with `--vocab`. Fewer candidates means fewer confusions.
4. Accept the best listed word only if all three hold:
   - `_none_` is not the top class;
   - the word holds at least **50%** of the probability among listed words;
   - its raw probability over all 2,304 words is at least **0.2**.

   The raw floor matters. Without it, a sign *outside* the list put its probability on the unlisted word it really
   is. After re-scaling over the list, the nearest listed word could look confident, which was the "it keeps saying
   DOG" problem. On validation clips the floor cut false words for out-of-list signs from 47% to about 19%, at a
   cost of about 2 points on listed signs. `--min-raw 0` restores the old behaviour.
5. Greedy non-maximum suppression: take the most confident window, drop windows that overlap it by more than
   **20%**, repeat.
6. Sort the kept windows by time and merge identical neighbours (the same sign seen twice).

The thresholds were tuned on validation clips, not on the test set.

**Word mode** (`recognize_word`, chosen per connection with `set_mode`) treats the whole segment as one sign. It
returns only the single best accepted window, plus the three likeliest words as alternatives. The segmenter cuts after
9 idle frames (~0.6 s) instead of 21, and the composer is skipped.

## 4. Sentence boundaries (`src/serve/segmenter.py`)

`ContinuousSignSegmenter` decides when a sentence is finished. A frame counts as **idle** when:

- no hand is visible, or
- the wrists have dropped well below their signing height, or
- there is almost no motion.

The first active frame starts a sentence; a short pre-roll buffer keeps the frames just before it. After
**21 idle frames in a row** (~1.4 s) the sentence is cut (`pause_detected`) and the trailing idle frames are
removed. Segments shorter than 10 frames are discarded as twitches. A cut is forced at 300 frames
(`max_duration_reached`). While the user signs, the server sends `live_words` every 8 frames for instant feedback.

## 5. Gloss → English composer (`src/sign/composer.py`)

`GlossComposer` turns the word list into one English sentence with **Qwen2.5-1.5B-Instruct plus a LoRA adapter**
(`checkpoints/composer/lora_1.5b`). The adapter was fine-tuned on 24k gloss/English pairs; see
[TRAINING.md](TRAINING.md#5-sentence-composer). Notes:

- One word → returned directly, no model call ("HELLO" → "Hello.").
- Greedy decoding, at most 40 new tokens, about 0.4 s on the GPU.
- `adapter="auto"` (the default) uses the adapter when it exists. Otherwise the stock model runs with a few-shot
  prompt.
- The model is loaded only if it fits in free GPU memory. Otherwise it falls back to a smaller model, then to the CPU.

On the 115-sentence held-out set the fine-tuned 1.5B scored better than the stock 3B model on every measure.

### Faithfulness check (`src/sign/faithful.py`)

Every sentence is checked before it is shown:

- Is every signed word present (lemmas and aliases such as ME → I/my)?
- Was any content word added that was not signed? Grammar words such as "the" or "is" are allowed.

If either check fails, the dashboard shows the recognised words instead ("Dog, cat, bird."). The system therefore
never displays a fact the user did not sign. The same scorer is used by `scripts/eval_composer.py`.

## 6. Server (`src/serve/server.py`)

FastAPI app serving the dashboard, the REST endpoints and `/ws/translate` ([API.md](API.md)). On start-up it
picks the model:

- **Sign models:** `checkpoints/sign/final_custom` if it exists (your own fine-tune), else `checkpoints/sign/final`.
- **Word list:** `--vocab` file, `all`, or the default demo list.
- **Composer:** fine-tuned 1.5B when the adapter exists.
- **Legacy model:** `--model-type t5` still runs the old How2Sign T5 model.

Recognition runs in a worker thread, so the event loop keeps streaming.

## 7. Resource safety (`src/sign/resources.py`)

The PC is shared with other users. `free_ram_gb()`, `free_vram_gb()` and `safe_workers()` are used by the training and
extraction scripts. They wait for free memory, size data-loader workers to the available RAM and run at low priority.
