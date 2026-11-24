# ASL Translator — real-time American Sign Language to English

A webcam application that recognises American Sign Language (ASL) signs and turns them into English sentences, live,
in the browser. The camera image never leaves the user's device: the browser extracts body and hand landmarks with
MediaPipe and streams only those numbers to the server.

```
camera ─► MediaPipe (browser) ─► landmarks ─► sign classifier ─► words ─► Qwen 2.5 (fine-tuned) ─► English sentence
                                   1200 numbers/frame   2,304 known signs     "ME GO SCHOOL TOMORROW"   "I will go to school tomorrow."
```

## Results (people the model never saw during training)

| | |
|---|---|
| Words the system knows | 2,304 (366-word everyday demo list active by default) |
| Correct word, demo list | **89.4%** top-1 · 98.4% top-5 |
| Correct word, all 2,304 words | 76.1% top-1 · 94.6% top-5 |
| 3 signs in a row through the live server | 84% of words found · 70% of sequences exactly right |
| Gloss → English sentence (115 held-out sentences) | 99.8% of signed words kept · 0% invented content · 99.1% correct question/statement form |

Full numbers and how they were measured: [reports/restart_report.md](reports/restart_report.md).

## Quick start

Needs Linux, Python 3.10, 16 GB RAM, ~15 GB disk, and ideally an NVIDIA GPU with ≥ 6 GB free (it also runs on CPU,
just slower). Details: [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md).

```bash
# 1. environment (Python 3.10, NVIDIA GPU recommended)
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader wordnet

# 2. start the server (loads the sign models + the Qwen composer, ~30 s)
python -m src.serve.server --port 8000

# 3. open http://localhost:8000 and click "Start Camera"
```

To use it from a phone or another laptop, share it with a Cloudflare tunnel (HTTPS, so the camera works):

```bash
~/.local/bin/cloudflared tunnel --edge-ip-version 4 --url http://localhost:8000
# prints https://<random-name>.trycloudflare.com
```

The trained models live in `checkpoints/` and the datasets in `data/`; both are git-ignored. A fresh clone needs them
copied over, or rebuilt with [docs/TRAINING.md](docs/TRAINING.md).

## Documentation

| Document | What it covers |
|---|---|
| [docs/PROJECT_EXPLAINED.md](docs/PROJECT_EXPLAINED.md) | **Start here to learn the project:** how it was built, every concept with what to search on YouTube, code walkthrough, likely exam questions |
| [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) | Hardware, software, files and network needed to run, use and train the project |
| [docs/DEMO_GUIDE.md](docs/DEMO_GUIDE.md) | Running and sharing the app, how to sign for best results, troubleshooting |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | How each stage works: landmarks, classifier, decoding, segmentation, composer, safety check |
| [docs/TRAINING.md](docs/TRAINING.md) | Rebuilding the datasets and models, evaluation scripts, working on the shared PC |
| [docs/CLOUD.md](docs/CLOUD.md) | Testing Qwen 3B on free cloud GPUs (Kaggle/Colab notebook), student-offer facts, `--composer` choices |
| [docs/API.md](docs/API.md) | HTTP endpoints and the WebSocket message protocol |
| [reports/restart_report.md](reports/restart_report.md) | Why the original system failed, what was changed, all results |

## Project layout

```
app/                 browser dashboard (index.html, static/js/landmarks.js, socket.js, ui.js)
src/extract/         MediaPipe Holistic wrapper + landmark normalisation (Python twin of landmarks.js)
src/sign/            current system: data.py, model.py, train.py, recognizer.py, composer.py, faithful.py, resources.py
src/serve/           FastAPI server, WebSocket endpoint, sentence segmenter
scripts/             dataset extraction, training pipelines, evaluation, recording your own signs
tests/               pytest suite (normalisation, JS/Python parity, server)
data/                datasets, landmarks, word lists, sentence data          (git-ignored)
checkpoints/         sign/final (deployed ensemble), composer/lora_1.5b      (git-ignored)
reports/             phase reports; restart_report.md is the current one
```

`src/models`, `src/train`, `src/eval`, `src/datasets`, `configs/`, `instructions.md` and `plan.md` belong to the first
version of the project (How2Sign sentence translation with a T5 decoder). They are kept for reference and can still be
run with `--model-type t5`; [reports/restart_report.md](reports/restart_report.md) explains why that approach was replaced.

## Tests

```bash
pytest tests                                   # unit + parity + server tests (28)
python scripts/ws_test.py 30                   # real WebSocket server on unseen-signer clips
python scripts/eval_vocab.py checkpoints/sign/final
python scripts/eval_composer.py --adapter checkpoints/composer/lora_1.5b
```

## Limitations

- Accuracy is measured on recorded test clips. Live results depend on lighting, framing and signing speed.
- It recognises individual signs, not fingerspelling or full ASL grammar. Facial grammar (eyebrows for questions) is
  approximated: "YOU …" without a question word is read as a yes/no question.
- Pausing with hands down for more than ~1.4 s ends the sentence.
