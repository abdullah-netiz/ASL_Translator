"""
src/serve/server.py — FastAPI + WebSocket Real-Time Translation Server.

Provides:
  - HTTP endpoints for health check, clip simulation, and static asset serving
  - WebSocket endpoint `/ws/translate` for streaming normalized landmark frames
  - Automatic sentence segmentation via ContinuousSignSegmenter
  - Real-time ASL-to-English translation via ASLInferencePipeline
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.serve.inference import ASLInferencePipeline
from src.serve.segmenter import ContinuousSignSegmenter

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger("server")

app = FastAPI(
    title="ASL Real-Time Translation Server",
    description="Real-Time Continuous ASL-to-English Translation using MediaPipe Holistic and T5",
    version="1.0.0",
)

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline and config holder
pipeline: ASLInferencePipeline | None = None
server_config: dict[str, Any] = {}


def get_pipeline() -> ASLInferencePipeline:
    global pipeline
    if pipeline is None:
        model_type = server_config.get("model_type", "t5")
        checkpoint = server_config.get("checkpoint", None)
        cfg_path = server_config.get("config", None)
        beam_size = server_config.get("beam_size", 5)
        pipeline = ASLInferencePipeline(
            model_type=model_type,
            config_path=cfg_path,
            checkpoint_path=checkpoint,
            beam_size=beam_size,
        )
    return pipeline


# ---------------------------------------------------------------------------
# HTTP Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    p = get_pipeline()
    return {
        "status": "healthy",
        "model_type": p.model_type,
        "device": str(p.device),
        "beam_size": p.beam_size,
    }


@app.get("/api/status")
async def get_status() -> dict[str, Any]:
    """Returns server and model status."""
    p = get_pipeline()
    return {
        "status": "ready",
        "model_type": p.model_type,
        "device": str(p.device),
        "beam_size": p.beam_size,
        "checkpoint": str(p.checkpoint_path),
    }


@app.get("/api/sample-clips")
async def list_sample_clips() -> dict[str, Any]:
    """Returns a list of pre-extracted test/val clips for simulation in browser."""
    val_dir = _REPO_ROOT / "data" / "landmarks" / "val"
    test_dir = _REPO_ROOT / "data" / "landmarks" / "test"

    val_clips = [p.name for p in sorted(val_dir.glob("*.npy"))[:15]] if val_dir.exists() else []
    test_clips = [p.name for p in sorted(test_dir.glob("*.npy"))[:15]] if test_dir.exists() else []

    return {
        "val_clips": val_clips,
        "test_clips": test_clips,
    }


@app.get("/api/clip/{split}/{clip_name}")
async def get_sample_clip(split: str, clip_name: str) -> Any:
    """Returns frames of a sample clip formatted as JSON array for client simulation."""
    if split not in ("val", "test", "train"):
        raise HTTPException(status_code=400, detail="Invalid split")

    clip_path = _REPO_ROOT / "data" / "landmarks" / split / clip_name
    if not clip_path.exists():
        raise HTTPException(status_code=404, detail="Clip not found")

    arr = np.load(clip_path).astype(np.float32)
    return {
        "clip_name": clip_name,
        "split": split,
        "num_frames": arr.shape[0],
        "feature_dim": arr.shape[1],
        "frames": arr.tolist(),
    }


@app.post("/api/translate")
async def translate_direct(payload: dict[str, Any]) -> dict[str, Any]:
    """Direct REST translation endpoint for batch or test evaluation."""
    p = get_pipeline()
    if "landmarks" not in payload:
        raise HTTPException(status_code=400, detail="Missing 'landmarks' in request body")

    lm_arr = np.array(payload["landmarks"], dtype=np.float32)
    beam_size = payload.get("beam_size", p.beam_size)
    result = p.translate(lm_arr, beam_size=beam_size)
    return result


# ---------------------------------------------------------------------------
# WebSocket Endpoint: /ws/translate
# ---------------------------------------------------------------------------

@app.websocket("/ws/translate")
async def websocket_translate(websocket: WebSocket) -> None:
    """
    Real-time streaming translation over WebSocket.
    Client sends landmark frames (JSON or Binary Float32Array).
    Server processes continuous stream, segments sentences, and pushes translations.
    """
    await websocket.accept()
    p = get_pipeline()
    segmenter = ContinuousSignSegmenter(
        min_frames=server_config.get("min_frames", 15),
        max_frames=server_config.get("max_frames", 240),
        pause_frames=server_config.get("pause_frames", 12),
        motion_threshold=server_config.get("motion_threshold", 0.5),
        wrist_drop_threshold=server_config.get("wrist_drop_threshold", 1.8),
    )

    logger.info("WebSocket client connected to /ws/translate")
    await websocket.send_json({
        "type": "connected",
        "model_type": p.model_type,
        "device": str(p.device),
        "message": "Connected to ASL Translation WebSocket server.",
    })

    frame_counter = 0

    try:
        while True:
            # Receive either binary or text frame
            message = await websocket.receive()

            if "bytes" in message and message["bytes"] is not None:
                # Binary mode: 1200 32-bit floats = 4800 bytes
                raw_bytes = message["bytes"]
                expected_len = 1200 * 4
                if len(raw_bytes) != expected_len:
                    # Invalid frame byte size
                    continue
                frame = np.frombuffer(raw_bytes, dtype=np.float32)
            elif "text" in message and message["text"] is not None:
                try:
                    data = json.loads(message["text"])
                except Exception:
                    continue

                msg_type = data.get("type", "frame")

                if msg_type == "ping":
                    await websocket.send_json({"type": "pong", "time": time.time()})
                    continue
                elif msg_type == "reset":
                    segmenter.reset()
                    await websocket.send_json({"type": "reset_ack", "message": "Segmenter reset"})
                    continue
                elif msg_type == "force_cut":
                    # Force translation of currently buffered frames
                    if len(segmenter.current_frames) >= 5:
                        seg_arr = np.array(segmenter.current_frames, dtype=np.float32)
                        segmenter.reset()
                        t_res = p.translate(seg_arr)
                        await websocket.send_json({
                            "type": "translation",
                            "text": t_res["text"],
                            "tokens": t_res["tokens"],
                            "num_tokens": t_res["num_tokens"],
                            "num_frames": t_res["num_frames"],
                            "latency_ms": t_res["latency_ms"],
                            "fps_processed": t_res["fps_processed"],
                            "reason": "force_cut",
                            "timestamp": time.time(),
                        })
                    continue
                elif msg_type == "frame":
                    frame = np.array(data.get("landmarks", []), dtype=np.float32)
                    if frame.shape != (1200,):
                        continue
                else:
                    continue
            else:
                continue

            frame_counter += 1

            # Process frame through online segmenter
            segment_ready, segment_data, info = segmenter.process_frame(frame)

            # Send periodic state update back to client (every 3 frames)
            if frame_counter % 3 == 0:
                await websocket.send_json({
                    "type": "status",
                    "is_signing": info["is_signing"],
                    "buffered_frames": info["buffered_frames"],
                    "pause_counter": info["pause_counter"],
                    "vel_norm": info["vel_norm"],
                })

            # When a sentence boundary is detected, translate immediately!
            if segment_ready and segment_data is not None:
                logger.info(
                    "Sentence boundary cut (%s): Translating %d frames...",
                    info["cut_reason"], len(segment_data)
                )

                # Send translating notification
                await websocket.send_json({
                    "type": "translating",
                    "num_frames": len(segment_data),
                    "reason": info["cut_reason"],
                })

                # Run translation (non-blocking for event loop)
                t_res = await asyncio.to_thread(p.translate, segment_data)

                logger.info(
                    "Translated in %.1f ms: '%s'",
                    t_res["latency_ms"], t_res["text"]
                )

                # Push translation result to dashboard
                await websocket.send_json({
                    "type": "translation",
                    "text": t_res["text"],
                    "tokens": t_res["tokens"],
                    "num_tokens": t_res["num_tokens"],
                    "num_frames": t_res["num_frames"],
                    "latency_ms": t_res["latency_ms"],
                    "fps_processed": t_res["fps_processed"],
                    "reason": info["cut_reason"],
                    "timestamp": time.time(),
                })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.error("WebSocket error: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Static Web App Mount
# ---------------------------------------------------------------------------

app_dir = _REPO_ROOT / "app"
static_dir = app_dir / "static"

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def serve_index() -> FileResponse:
    index_file = app_dir / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(str(index_file))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ASL Real-Time Translation Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--model-type", default="t5", choices=["t5", "transformer", "lstm"], help="Model type")
    parser.add_argument("--config", default="configs/t5_decoder.yaml", help="Path to model config")
    parser.add_argument("--checkpoint", default="checkpoints/t5/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam width for translation")
    parser.add_argument("--min-frames", type=int, default=15, help="Min frames for sentence boundary")
    parser.add_argument("--pause-frames", type=int, default=12, help="Pause frames to cut sentence")
    args = parser.parse_args()

    server_config["model_type"] = args.model_type
    server_config["config"] = args.config
    server_config["checkpoint"] = args.checkpoint
    server_config["beam_size"] = args.beam_size
    server_config["min_frames"] = args.min_frames
    server_config["pause_frames"] = args.pause_frames

    # Pre-load pipeline to verify checkpoint and warm up GPU before listening
    logger.info("Pre-warming ASLInferencePipeline...")
    _ = get_pipeline()
    logger.info("Pipeline ready. Starting Uvicorn on %s:%d...", args.host, args.port)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
