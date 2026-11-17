# Real-Time ASL Translation Web Dashboard

A browser-based real-time continuous American Sign Language (ASL) to English translation application powered by **MediaPipe Holistic** landmark extraction in JavaScript, a **FastAPI + WebSocket** server, online **temporal sentence segmentation**, and a **Transformer + Pretrained T5** translation engine.

---

## Architecture Overview

```
 [Webcam Video / Simulator]
            │
            ▼
 [MediaPipe Holistic JS] ────► Extracts Pose (17), Hands (2x21), Face (59)
            │
            ▼
 [LandmarkNormalizer (JS)] ──► Shoulder-centered, wrist-relative, velocity delta (1200 floats)
            │
            ▼ (Binary Float32Array over WebSocket)
 [/ws/translate (FastAPI)]
            │
            ▼
 [ContinuousSignSegmenter] ──► Detects pauses, wrist dropping, and sentence boundaries
            │ (On boundary cut)
            ▼
 [ASLInferencePipeline]   ──► Transformer Landmark Encoder + Pretrained T5-small Decoder
            │
            ▼
 [Translation Results]     ──► Streamed back to Dashboard UI (Current Banner + Transcript)
```

---

## Features

1. **Client-Side Landmark Extraction (Zero Video Upload)**:
   - Video is processed directly in the user's browser via MediaPipe Holistic JS.
   - Raw video frames never leave the client's machine, ensuring privacy and minimal network bandwidth.
   - Bit-exact mathematical parity with `src/extract/normalize.py` verified by `tests/test_js_python_parity.py`.

2. **Online Sentence Boundary Detection (`ContinuousSignSegmenter`)**:
   - Automatically cuts continuous signing into discrete sentences based on hand presence, wrist elevation drops, and low-motion pauses.
   - No manual start/stop button clicks required during continuous natural signing.

3. **Sub-2-Second End-to-End Latency**:
   - Transformer Landmark Encoder + learned projection + T5 text decoder runs in ~1.0s on GPU with beam width 5 (and ~120ms with greedy search).
   - Real-time compliance verified against the $< 2.0$s target.

4. **Webcam and Simulation Modes**:
   - **Live Webcam**: Real-time sign language capture directly from your webcam with landmark visualization canvas.
   - **Clip Simulator**: Select and stream pre-extracted How2Sign validation/test clips frame-by-frame at 15 fps to evaluate translation without requiring a live camera.

---

## Launching the Application

### 1. Start the Server

Using the `Makefile` shortcut:
```bash
make serve
```

Or run the CLI directly:
```bash
.venv/bin/python src/serve/server.py --port 8000 --model-type t5
```

Options:
- `--host`: Host interface to bind (default `0.0.0.0`)
- `--port`: Port number (default `8000`)
- `--model-type`: `t5`, `transformer`, or `lstm` (default `t5`)
- `--checkpoint`: Path to model weights (default `checkpoints/t5/best_model.pt`)
- `--beam-size`: Beam search width (default `5`)
- `--pause-frames`: Number of rest frames to trigger sentence boundary (default `12`)

### 2. Open in Browser

Open Google Chrome or any modern WebGL-compatible browser:
```
http://localhost:8000
```

---

## Testing & Parity Verification

Run parity and pipeline tests:
```bash
.venv/bin/pytest -v tests/test_js_python_parity.py
```
