# System requirements

There are three different jobs, with very different needs:

1. **Running the app (the server):** the PC that runs `python -m src.serve.server`.
2. **Using the app (a client device):** any laptop or phone that opens it in a browser.
3. **Training the models:** only needed to rebuild or improve the models.

Figures marked *measured* come from the development machine (RTX 4000 Ada, 28 cores, 30 GB RAM, Ubuntu 26.04).

## 1. Server: running the translator

| | Minimum (CPU only) | Recommended (GPU) |
|---|---|---|
| CPU | 4 cores, x86-64 | 8+ cores |
| RAM | 16 GB (peak ~10 GB used, *measured*) | 16 GB (server uses ~1.5 GB, *measured*) |
| GPU | none | NVIDIA, **≥ 6 GB free VRAM** (server uses 4.6 GB, *measured*) |
| Disk | 15 GB free | 15 GB free (SSD) |
| Sentence latency | ~1.4–2 s (*measured*, 8 threads) | ~0.4–0.5 s (*measured*) |

- **GPU memory:**
  - The sign models need under 0.5 GB.
  - The Qwen 1.5B sentence model needs about 4 GB and loads on the GPU only if that much is free.
  - Otherwise it falls back to the CPU on its own. The app still works, just slower.
- **What uses the disk:**
  - Python environment: ~6.6 GB.
  - Qwen 2.5 1.5B download: 2.9 GB.
  - Sign models: 22 MB.
  - Fine-tuned adapter: 71 MB.
  - Word lists and sample clips: under 1.1 GB.
- **CPU-only mode** is fine for a demo, but sentences appear 1–2 s after you stop signing instead of about 0.5 s.

### Software

| Component | Version | Notes |
|---|---|---|
| OS | Linux, 64-bit | Tested on Ubuntu 26.04. Windows/macOS untested; macOS has no CUDA, so it would run in CPU mode |
| Python | **3.10** (3.11 at most) | `mediapipe==0.10.14` does not support newer Python |
| NVIDIA driver | ≥ 525 | Needed for the CUDA 12.1 build of PyTorch (tested with 620.70). The CUDA toolkit itself is **not** needed |
| Python packages | `requirements.txt` | Includes torch 2.3.1 (CUDA 12.1), transformers 4.42.4, peft 0.11.1, mediapipe 0.10.14, FastAPI |
| NLTK data | `wordnet` | `python -m nltk.downloader wordnet`. Used by the sentence safety check, which falls back to a simpler method without it |
| cloudflared | any recent | Optional, for sharing to other devices ([DEMO_GUIDE.md](DEMO_GUIDE.md)) |

### Files the server needs

These are git-ignored, so copy them from the development PC or rebuild them ([TRAINING.md](TRAINING.md)):

| Path | Size | Required? |
|---|---|---|
| `checkpoints/sign/final/m0.pt … m2.pt` | 22 MB | Yes: the sign recognisers |
| `checkpoints/composer/lora_1.5b/` | 71 MB | Recommended: fine-tuned sentence model (otherwise stock Qwen, less accurate) |
| `data/aslcitizen/vocab_demo.txt` | < 1 MB | Recommended: default word list (otherwise all 2,304 words are active) |
| `data/wlasl/landmarks/`, `data/wlasl/manifest.csv` | ~1 GB | Optional: clips for the dashboard's *Simulate* feature |
| Qwen2.5-1.5B-Instruct | 2.9 GB | Downloaded from Hugging Face on first start into `~/.cache/huggingface/` |

### Network

- **First start:** internet access to download Qwen from Hugging Face. After that the server runs offline.
- **Port:** 8000 by default (`--port` to change). The server listens on all interfaces.
- **Sharing outside the local network:** a tunnel such as Cloudflare, which gives HTTPS. Browsers allow the camera
  only on `localhost` or HTTPS.

## 2. Client devices: using the app

| | Requirement |
|---|---|
| Browser | Recent Chrome, Edge, Firefox or Safari with WebGL and WebAssembly (any browser from the last ~3 years) |
| Device | Laptop with a dual-core CPU or better, or a recent phone. MediaPipe runs on the device, so very old phones will be slow |
| Camera | Built-in webcam or phone camera; 720p is plenty. An Android phone running *IP Webcam* also works |
| Connection | **HTTPS link** (tunnel) or `http://localhost` on the server itself |
| Internet | Needed: the page loads MediaPipe from the jsDelivr CDN |
| Bandwidth | ~0.6 Mbit/s upload while signing (1,200 numbers per frame × 15 fps); no video is uploaded |

Physical set-up: good light on face and hands, upper body and both hands in frame, about 1–1.5 m from the camera.

## 3. Training the models

Only needed to rebuild or improve the models. Times are measured on the development machine with the GPU shared
with other users. Classifier GPU memory (~2 GB) is the level the scripts wait for, not a profiled peak.

| Task | GPU memory | RAM | Disk | Time (*measured*) |
|---|---|---|---|---|
| Landmark extraction (ASL Citizen, 6 workers) | none (CPU) | ~4 GB | 43 GB zip + 8 GB landmarks | ~9.5 h |
| Landmark extraction (WLASL) | none (CPU) | ~3 GB | 5.1 GB videos + 1 GB landmarks | ~1 h (estimate) |
| Sign classifier: one model | ~2 GB | 8 GB free | small | 30–75 min |
| Sign classifier: full sweep + ensemble (`citizen_pipeline.sh`) | ~2 GB | 8 GB free | small | ~5.5 h |
| Train+val retrain, 3 models (`citizen_trainval.sh`) | ~2 GB | 8 GB free | small | ~2.7 h |
| Sentence model LoRA (`train_composer.py`) | **~4.5 GB** | 8 GB | 3 GB (Qwen) | ~70 min |
| Personal fine-tune (`finetune_custom.sh`) | ~2 GB | 8 GB free | small | ~3 h |

- **Total disk for a full rebuild:** about 100 GB. The current `data/` folder is 95 GB, including the legacy How2Sign
  data. 150 GB free is comfortable.
- **CPU:** extraction scales with cores; 8+ cores recommended. The scripts size their worker count from free RAM.
- **Shared GPU:**
  - The training scripts wait for enough free memory and retry after an out-of-memory crash.
  - Stopping the demo server frees ~4.6 GB.
  - Check first with `nvidia-smi` and `free -g`.

## Quick check

```bash
python --version                                   # 3.10.x
nvidia-smi --query-gpu=name,memory.free,driver_version --format=csv
free -g | sed -n 2p                                # available RAM
df -h .                                            # free disk
python -c "import torch; print(torch.cuda.is_available())"
```
