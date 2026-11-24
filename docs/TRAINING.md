# Training and evaluation

How to rebuild every model from the raw datasets and re-measure it. The deployed models were produced exactly this way.
All commands run from the project root with the virtual environment active.

> **Shared PC.** Other users' jobs often hold most of the GPU memory. All long-running scripts below run at
> low priority, size their workers from free RAM, wait until enough RAM/VRAM is free, and retry after a CUDA
> out-of-memory crash. Check before starting anything big: `free -g` and `nvidia-smi`. Our own demo server holds
> ~4 GB of GPU memory; stopping it frees room for training.

## 1. Datasets

| Dataset | Location | Content |
|---|---|---|
| WLASL v0.3 | `data/archive/` (`videos/`, `WLASL_v0.3.json`) | Isolated signs, mixed sources; official signer-disjoint splits |
| ASL Citizen | `data/aslcitizen/ASL_Citizen.zip` | 83,399 webcam clips, 52 signers, 2,303 words; official splits |
| How2Sign | `data/raw/` | Sentence videos + English; used only by the legacy T5 model |

Only words with at least 10 training clips are used: **2,304 words** (all 2,303 ASL Citizen words + "trade" from WLASL). That gives 46,300 train / 11,915 validation /
34,013 test clips. Test signers never appear in training or validation.

## 2. Landmark extraction

Both extractors use the same MediaPipe wrapper and normalisation as the live app (15 fps, 1,200 numbers per frame).

```bash
python scripts/wlasl_extract.py 4          # -> data/wlasl/landmarks/*.npy + data/wlasl/manifest.csv   (11,980 clips)
python scripts/aslcitizen_extract.py 6     # -> data/aslcitizen/landmarks/*.npy + manifest.csv         (~7 h)
```

- WLASL clips are cropped to the annotated signer box and trimmed to the annotated frames.
- ASL Citizen is read straight from the zip. The extractor can resume after a stop, and workers pause while free
  RAM is below 4 GB.
- Each ASL Citizen word is named by its most common English spelling in the file names. The CSV gloss names drop
  spaces.

Word lists:

| File | Words | Use |
|---|---|---|
| `data/aslcitizen/all_words.txt` | 2,303 | Every ASL Citizen word (the model also knows "trade" from WLASL) |
| `data/aslcitizen/vocab_demo.txt` | 366 | Default active list (everyday + conversational) |
| `data/aslcitizen/vocab_everyday.txt` | 365 | Everyday words |
| `data/wlasl/vocab_convo.txt` | 141 | Original conversational list |

## 3. Sign classifier

`src/sign/train.py` trains one model. The important flags:

| Flag | Meaning |
|---|---|
| `--sources wlasl,citizen --min-train 10 --top-k 100000` | Both datasets, every word with ≥ 10 training clips |
| `--feat hands --derived` | Drop face/head features, add 146 hand-shape features |
| `--mirror 0.5` | Mirror half the training clips (left-handed signers, mirrored cameras) |
| `--d-model --layers --dropout --wd` | Model size and regularisation |
| `--epochs --patience` | Stop when validation loss has not improved for `patience` epochs |
| `--trainval` | Train on train+val for a fixed number of epochs (deployment); test is still untouched |
| `--custom` | Add your own recordings from `data/wlasl/custom/` (×4 oversampled) |

Training also uses:

- Label smoothing 0.1, AdamW with a cosine schedule, mixed precision.
- Augmentation: rotation, scale and shift of present points only; temporal crop and speed change; dropping a hand.
- 10% synthetic `_none_` examples.

The checkpoint saved is the one with the **lowest validation loss**. Each run writes `best.pt` and `results.json`
(best epoch, val loss, test accuracy).

### The pipeline that produced the deployed model

```bash
bash scripts/citizen_pipeline.sh     # sweep 4 configs -> pick by VAL loss -> 2 more seeds -> evaluate   (~6 h)
bash scripts/citizen_trainval.sh     # best config on train+val, 3 seeds x 100 epochs -> compare on test (~3 h)
```

Sweep results, all early-stopped on validation loss:

| Config | d_model | Layers | Dropout | Weight decay | Best val loss | Val top-1 |
|---|---|---|---|---|---|---|
| c1 | 256 | 2 | 0.2 | 0.05 | 1.013 | 75.4% |
| c2 | 256 | 4 | 0.2 | 0.05 | 1.050 | 74.6% |
| c3 | 384 | 4 | 0.25 | 0.05 | 1.078 | 74.9% |
| **c4** | **192** | **2** | **0.3** | **0.1** | **0.918** | **78.4%** |

The smallest, most regularised model won; bigger models overfit sooner. The early-stopped seeds peaked at
epochs 85–117, so the train+val retrain uses a fixed 100 epochs. At the chosen epoch, augmented training loss
(1.56) stayed above validation loss (0.93), which shows no overfitting.

### Deploying a new sign model

```bash
mv checkpoints/sign/final checkpoints/sign/final_backup_$(date +%F)
mkdir checkpoints/sign/final && cp checkpoints/sign/cit_tv_ens/m*.pt checkpoints/sign/final/
# restart the server
```

Kept backups: `final_wlasl141` (original 141-word model), `final_cit_trainonly` (train-only ensemble).

## 4. Evaluating word recognition

```bash
python scripts/eval_vocab.py checkpoints/sign/final --lists data/aslcitizen/vocab_demo.txt
python scripts/e2e_test.py 100        # 3 test clips chained per stream, recognizer called directly
python scripts/ws_test.py 30          # same idea through the real WebSocket server + segmenter
```

| Test-signer accuracy | Top-1 | Top-5 |
|---|---|---|
| 366 demo words, ASL Citizen | 89.4% | 98.4% |
| 366 demo words, WLASL | 81.9% | 97.4% |
| 141 conversational words, WLASL | 88.9% | 98.5% |
| All 2,304 words | 76.1% | 94.6% |

Chained streams: 85% word recall and 56% exact sequences (direct). Through the server: 84% recall and 70% exact.

## 5. Sentence composer

### Data

```bash
python scripts/make_sentence_data.py   # -> data/sentences/train.tsv (22,828) + val.tsv (1,201)
```

The generator is a small English grammar over the vocabulary. It covers:

- tense from time words;
- WH-word last;
- `NOT` before the verb;
- "YOU …" yes/no questions;
- plurals, possessives, "can", "want to".

Any pair whose gloss appears in the held-out test set `data/sentences/eval_gloss_en.tsv` is removed. That test set
has 115 hand-written sentences.

### Fine-tuning

```bash
python scripts/train_composer.py --bs 8 --accum 4 --min-vram 4.5     # ~70 min, needs ~4.5 GB free VRAM
```

- LoRA r = 16 (alpha 32, dropout 0.05) on all attention and MLP projections of Qwen2.5-1.5B-Instruct. That is
  18.5M trainable parameters (1.2%).
- Loss is computed only on the English answer.
- Learning rate 2e-4 with cosine decay, effective batch 32, up to 3 epochs.
- The adapter with the lowest validation loss is saved to `checkpoints/composer/lora_1.5b`. Training stops early
  after 4 evaluations without improvement.
- Micro-batches of 8 are needed because Qwen's 151k-token output layer is large.

### Evaluation

```bash
python scripts/eval_composer.py --adapter checkpoints/composer/lora_1.5b --show 20
python scripts/eval_composer.py --model Qwen/Qwen2.5-3B-Instruct --device cpu    # stock model for comparison
```

| Composer | Signed words kept | Sentences with added content | Question form | chrF |
|---|---|---|---|---|
| Qwen2.5-1.5B few-shot | 98.8% | 5.2% | 92.2% | 74.2 |
| Qwen2.5-3B few-shot | 98.3% | 4.3% | 97.4% | 81.0 |
| **Qwen2.5-1.5B + LoRA** | **99.8%** | **0.0%** | **99.1%** | **87.7** |

The evaluator scores the raw model output; the runtime faithfulness fallback is switched off for evaluation.
The test sentences and the generator were written by the same people, so these scores are probably optimistic for
free-form signing.

## 6. Personal fine-tuning

```bash
python scripts/record_signs.py hello thank_you water --n 8     # saves data/wlasl/custom/<word>__NN.npy
bash scripts/finetune_custom.sh                                # -> checkpoints/sign/final_custom (~3 h)
```

The server uses `final_custom` automatically when it exists. Delete that folder to return to the standard model.
