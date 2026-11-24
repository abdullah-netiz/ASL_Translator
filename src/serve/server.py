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
import csv
import json
import logging
import os
import re
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
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
DEFAULT_IPCAM_URL = "http://172.15.45.93:8080/video"
pipeline: ASLInferencePipeline | None = None
_pipeline_lock = threading.Lock()
server_config: dict[str, Any] = {}
DEMO_VOCAB = _REPO_ROOT / "data" / "aslcitizen" / "vocab_demo.txt"


def _active_vocab() -> str | None:
    """Word list the live recognizer is restricted to: --vocab file, 'all' = full model vocabulary,
    default = the curated everyday/conversational demo list (higher accuracy) when present."""
    v = server_config.get("vocab")
    if v == "all":
        return None
    if v:
        return v
    return str(DEMO_VOCAB) if DEMO_VOCAB.exists() else None


def normalize_ipcam_url(raw_url: str) -> str:
    """
    Normalizes user-supplied IP Webcam addresses into a valid HTTP(S) stream URL.
    Examples:
      - '172.15.45.93.8080'          -> 'http://172.15.45.93:8080/video'
      - '172.15.45.93:8080'          -> 'http://172.15.45.93:8080/video'
      - 'http://172.15.45.93:8080'   -> 'http://172.15.45.93:8080/video'
      - 'http://172.15.45.93:8080/shot.jpg' -> 'http://172.15.45.93:8080/shot.jpg'
    """
    url = (raw_url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="IP Webcam URL cannot be empty")

    # Fix common IPv4 + dot + port typo: e.g. 172.15.45.93.8080 -> 172.15.45.93:8080
    url = re.sub(
        r"^(https?://)?(\d{1,3}(?:\.\d{1,3}){3})\.(\d{2,5})(/.*)?$",
        r"\1\2:\3\4",
        url,
        flags=re.IGNORECASE,
    )

    if "://" not in url:
        url = f"http://{url}"

    parsed = urlparse(url)
    if parsed.scheme.lower() not in ("http", "https"):
        raise HTTPException(status_code=400, detail="Only http:// and https:// URLs are supported")
    if not parsed.hostname:
        raise HTTPException(status_code=400, detail="Invalid IP Webcam hostname or address")

    path = parsed.path if parsed.path and parsed.path != "/" else "/video"
    normalized = urlunparse((parsed.scheme.lower(), parsed.netloc, path, "", parsed.query, ""))
    return normalized


def get_pipeline() -> ASLInferencePipeline:
    global pipeline
    if pipeline is None:
        with _pipeline_lock:
            if pipeline is None:
                model_type = server_config.get("model_type") or ("sign" if (_REPO_ROOT / "checkpoints/sign/final").exists() else "t5")
                if model_type == "sign" and not server_config.get("sign_checkpoint") and (_REPO_ROOT / "checkpoints/sign/final_custom").exists():
                    server_config["sign_checkpoint"] = str(_REPO_ROOT / "checkpoints/sign/final_custom")
                checkpoint = server_config.get("checkpoint", None) if model_type != "sign" else server_config.get("sign_checkpoint")
                cfg_path = server_config.get("config", None)
                beam_size = server_config.get("beam_size", 5)
                if model_type == "sign":
                    from src.sign.recognizer import SignTranslator
                    pipeline = SignTranslator(checkpoint=checkpoint, composer_model=server_config.get("composer", "Qwen/Qwen2.5-3B-Instruct"), active_vocab=_active_vocab(), **({"min_raw": server_config["min_raw"]} if server_config.get("min_raw") is not None else {}))
                    return pipeline
                pipeline = ASLInferencePipeline(
                    model_type=model_type,
                    config_path=cfg_path,
                    checkpoint_path=checkpoint,
                    beam_size=beam_size,
                )
    return pipeline


async def get_pipeline_async() -> ASLInferencePipeline:
    if pipeline is not None:
        return pipeline
    return await asyncio.to_thread(get_pipeline)


# ---------------------------------------------------------------------------
# HTTP Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    p = await get_pipeline_async()
    return {
        "status": "healthy",
        "model_type": p.model_type,
        "device": str(p.device),
        "beam_size": p.beam_size,
    }


@app.get("/api/status")
async def get_status() -> dict[str, Any]:
    """Returns server and model status."""
    p = await get_pipeline_async()
    return {
        "status": "ready",
        "model_type": p.model_type,
        "device": str(p.device),
        "beam_size": p.beam_size,
        "checkpoint": str(p.checkpoint_path),
        "default_ipcam_url": server_config.get("ipcam_url", DEFAULT_IPCAM_URL),
    }


@app.get("/api/vocab")
async def get_vocab() -> dict[str, Any]:
    """Words the sign recogniser can currently understand (for the dashboard's help panel)."""
    p = await get_pipeline_async()
    words = list(getattr(p, "active_words", [g for g in getattr(p, "glosses", []) if g != "_none_"]))
    return {"count": len(words), "words": sorted(words)}


@app.get("/api/sample-clips")
async def list_sample_clips() -> dict[str, Any]:
    """Returns a list of pre-extracted test/val clips for simulation in browser."""
    wl = _wlasl_test_clips()
    if wl:
        return {"val_clips": wl, "test_clips": []}
    val_dir = _REPO_ROOT / "data" / "landmarks" / "val"
    test_dir = _REPO_ROOT / "data" / "landmarks" / "test"

    val_clips = [p.name for p in sorted(val_dir.glob("*.npy"))[:15]] if val_dir.exists() else []
    test_clips = [p.name for p in sorted(test_dir.glob("*.npy"))[:15]] if test_dir.exists() else []

    return {
        "val_clips": val_clips,
        "test_clips": test_clips,
    }


def _wlasl_test_clips(n_per_word: int = 1) -> list[str]:
    """Held-out WLASL test clips named '<gloss>__<video_id>.npy' for browser simulation."""
    try:
        from src.sign.data import load_manifest
        vocab = _REPO_ROOT / "data" / "wlasl" / "vocab_convo.txt"
        if not vocab.exists() or not (_REPO_ROOT / "checkpoints/sign/final").exists():
            return []
        glosses, split = load_manifest(9999, vocab=str(vocab))
        seen: dict[str, int] = {}
        out = []
        for vid, y in split["test"]:
            if seen.get(glosses[y], 0) < n_per_word:
                seen[glosses[y]] = seen.get(glosses[y], 0) + 1
                out.append(f"{glosses[y]}__{vid}.npy")
        return sorted(out)
    except Exception:
        return []


def _lookup_reference_sentence(split: str, clip_stem: str) -> str:
    """Looks up the ground-truth English reference sentence for a clip from manifests."""
    manifest_path = _REPO_ROOT / "data" / "manifests" / f"{split}_manifest.csv"
    if not manifest_path.exists():
        return ""
    try:
        with open(manifest_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("clip_id") == clip_stem:
                    return row.get("sentence", "")
    except Exception:
        pass
    return ""


@app.get("/api/clip/{split}/{clip_name}")
async def get_sample_clip(split: str, clip_name: str) -> Any:
    """Returns frames of a sample clip formatted as JSON array for client simulation."""
    if split not in ("val", "test", "train"):
        raise HTTPException(status_code=400, detail="Invalid split")

    if "__" in clip_name:   # WLASL word clip: '<gloss>__<video_id>.npy'
        gloss, vid = clip_name[:-4].split("__", 1)
        wpath = _REPO_ROOT / "data" / "wlasl" / "landmarks" / f"{vid}.npy"
        if not wpath.exists() or not vid.isalnum():
            raise HTTPException(status_code=404, detail="Clip not found")
        arr = np.load(wpath).astype(np.float32)
        return {"clip_name": clip_name, "split": split, "num_frames": arr.shape[0],
                "feature_dim": arr.shape[1], "reference_text": f"sign for '{gloss}'", "frames": arr.tolist()}

    clip_path = _REPO_ROOT / "data" / "landmarks" / split / clip_name
    if not clip_path.exists():
        raise HTTPException(status_code=404, detail="Clip not found")

    arr = np.load(clip_path).astype(np.float32)
    return {
        "clip_name": clip_name,
        "split": split,
        "num_frames": arr.shape[0],
        "feature_dim": arr.shape[1],
        "reference_text": _lookup_reference_sentence(split, clip_path.stem),
        "frames": arr.tolist(),
    }


@app.post("/api/translate")
async def translate_direct(payload: dict[str, Any]) -> dict[str, Any]:
    """Direct REST translation endpoint for batch or test evaluation."""
    p = await get_pipeline_async()
    if "landmarks" not in payload:
        raise HTTPException(status_code=400, detail="Missing 'landmarks' in request body")

    lm_arr = np.array(payload["landmarks"], dtype=np.float32)
    beam_size = payload.get("beam_size", p.beam_size)
    result = await asyncio.to_thread(p.translate, lm_arr, beam_size)
    return result


# ---------------------------------------------------------------------------
# IP Webcam Proxy Endpoints (eliminates browser CORS / tainted canvas issues)
# ---------------------------------------------------------------------------

@app.get("/api/ipcam/test")
async def test_ipcam_connection(
    url: str = Query(default=DEFAULT_IPCAM_URL, description="IP Webcam URL or host:port"),
) -> dict[str, Any]:
    """Tests connectivity to an IP Webcam stream and returns the normalized URL."""
    resolved_url = normalize_ipcam_url(url)
    parsed = urlparse(resolved_url)
    host_port = parsed.netloc

    timeout = httpx.Timeout(connect=3.5, read=4.5, write=3.5, pool=3.5)
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            async with client.stream("GET", resolved_url) as resp:
                if resp.status_code != 200:
                    return {
                        "ok": False,
                        "resolved_url": resolved_url,
                        "error": f"IP Webcam responded with HTTP {resp.status_code}. Ensure stream path is /video.",
                    }
                content_type = resp.headers.get("content-type", "application/octet-stream")
                return {
                    "ok": True,
                    "resolved_url": resolved_url,
                    "content_type": content_type,
                }
    except httpx.ConnectTimeout:
        logger.warning("IP Webcam connection timed out for %s", resolved_url)
        return {
            "ok": False,
            "resolved_url": resolved_url,
            "error": (
                f"Connection timed out to {host_port}. "
                "Ensure your phone is unlocked, IP Webcam app is open with 'Start server' active, "
                "and the IP on the phone screen matches."
            ),
        }
    except httpx.ConnectError as exc:
        logger.warning("IP Webcam connection failed for %s: %s", resolved_url, exc)
        return {
            "ok": False,
            "resolved_url": resolved_url,
            "error": (
                f"Could not reach {host_port}. "
                "Ensure your phone and PC are on the same Wi-Fi network and 'Start server' is running in IP Webcam."
            ),
        }
    except httpx.ReadTimeout:
        logger.warning("IP Webcam read timed out for %s", resolved_url)
        return {
            "ok": False,
            "resolved_url": resolved_url,
            "error": (
                f"Connected to {host_port}, but video stream timed out. "
                "Check camera permissions and keep the phone screen awake."
            ),
        }
    except Exception as exc:
        logger.warning("IP Webcam connectivity test failed for %s: %s", resolved_url, exc)
        err_msg = str(exc).strip() or type(exc).__name__
        return {
            "ok": False,
            "resolved_url": resolved_url,
            "error": f"Failed to reach {host_port}: {err_msg}",
        }


@app.get("/api/ipcam/stream")
async def proxy_ipcam_stream(
    request: Request,
    url: str = Query(default=DEFAULT_IPCAM_URL, description="IP Webcam MJPEG or JPEG URL"),
) -> StreamingResponse:
    """
    Proxies an IP Webcam MJPEG stream (e.g. http://172.15.45.93:8080/video)
    over same-origin HTTP so the browser can render it and run MediaPipe JS
    without CORS or tainted-canvas restrictions.
    """
    resolved_url = normalize_ipcam_url(url)
    timeout = httpx.Timeout(connect=5.0, read=None, write=5.0, pool=5.0)

    client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)
    try:
        req = client.build_request("GET", resolved_url)
        upstream = await client.send(req, stream=True)
    except Exception as exc:
        await client.aclose()
        raise HTTPException(status_code=502, detail=f"Failed to connect to IP Webcam ({resolved_url}): {exc}")

    if upstream.status_code != 200:
        status_code = upstream.status_code
        await upstream.aclose()
        await client.aclose()
        raise HTTPException(
            status_code=502,
            detail=f"IP Webcam ({resolved_url}) returned HTTP {status_code}",
        )

    content_type = upstream.headers.get("content-type", "multipart/x-mixed-replace; boundary=frame")

    # Case 1: Single JPEG snapshot endpoint (e.g. /shot.jpg) -> convert to MJPEG stream
    if content_type.lower().startswith("image/jpeg"):
        first_frame = await upstream.aread()
        await upstream.aclose()

        async def snapshot_mjpeg_generator():
            try:
                if first_frame:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + first_frame
                        + b"\r\n"
                    )
                while not await request.is_disconnected():
                    await asyncio.sleep(1.0 / 15.0)
                    resp = await client.get(resolved_url)
                    if resp.status_code == 200 and resp.content:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                            + resp.content
                            + b"\r\n"
                        )
            except (asyncio.CancelledError, httpx.HTTPError):
                pass
            finally:
                await client.aclose()

        return StreamingResponse(
            snapshot_mjpeg_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Access-Control-Allow-Origin": "*",
            },
        )

    # Case 2: Native MJPEG stream (e.g. /video, /videofeed) -> zero-overhead byte passthrough
    async def mjpeg_passthrough_generator():
        try:
            async for chunk in upstream.aiter_bytes(chunk_size=16384):
                if await request.is_disconnected():
                    break
                if chunk:
                    yield chunk
        except (asyncio.CancelledError, httpx.HTTPError):
            pass
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        mjpeg_passthrough_generator(),
        media_type=content_type,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/api/ipcam/snapshot")
async def proxy_ipcam_snapshot(
    url: str = Query(default=DEFAULT_IPCAM_URL, description="IP Webcam snapshot URL"),
) -> Response:
    """Fetches a single JPEG snapshot from the IP Webcam."""
    resolved_url = normalize_ipcam_url(url)
    # If caller passed /video, try /shot.jpg for single snapshot
    parsed = urlparse(resolved_url)
    if parsed.path in ("/video", "/videofeed"):
        resolved_url = urlunparse((parsed.scheme, parsed.netloc, "/shot.jpg", "", parsed.query, ""))

    timeout = httpx.Timeout(connect=3.0, read=5.0, write=3.0, pool=3.0)
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(resolved_url)
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail=f"IP Webcam returned HTTP {resp.status_code}")
            return Response(
                content=resp.content,
                media_type=resp.headers.get("content-type", "image/jpeg"),
                headers={
                    "Cache-Control": "no-cache",
                    "Access-Control-Allow-Origin": "*",
                },
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch IP Webcam snapshot: {exc}")


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
    p = await get_pipeline_async()
    is_sign = getattr(p, "model_type", "") == "sign"
    # Sign mode: a sentence = several signs separated by short pauses, so wait ~1.4 s of
    # idle (21 frames @15fps) before closing the utterance; the recogniser splits it into words.
    segmenter = ContinuousSignSegmenter(
        min_frames=server_config.get("min_frames") or (10 if is_sign else 12),
        max_frames=server_config.get("max_frames", 300 if is_sign else 240),
        pause_frames=server_config.get("pause_frames") or (21 if is_sign else 12),
        motion_threshold=server_config.get("motion_threshold", 0.08),
        wrist_drop_threshold=server_config.get("wrist_drop_threshold", 2.2),
    )

    # Recognition mode per connection: "sentence" (several signs -> composed English) or "word" (one sign -> that word,
    # short pause ends it, no sentence composer). The dashboard switches with {"type": "set_mode", "mode": "word"}.
    SENTENCE_PAUSE = segmenter.pause_frames
    WORD_PAUSE = server_config.get("word_pause_frames") or 9          # ~0.6 s @ 15 fps
    mode = "sentence"

    def run_translate(seg_arr: np.ndarray) -> dict[str, Any]:
        if is_sign and mode == "word":
            return p.translate_word(seg_arr)
        return p.translate(seg_arr)

    def result_msg(t_res: dict[str, Any], reason: str) -> dict[str, Any]:
        return {
            "type": "translation",
            "text": t_res["text"],
            "glosses": t_res.get("glosses", []),
            "alternatives": t_res.get("alternatives", []),
            "mode": mode,
            "tokens": t_res["tokens"],
            "num_tokens": t_res["num_tokens"],
            "num_frames": t_res["num_frames"],
            "latency_ms": t_res["latency_ms"],
            "fps_processed": t_res["fps_processed"],
            "reason": reason,
            "timestamp": time.time(),
        }

    logger.info("WebSocket client connected to /ws/translate")
    await websocket.send_json({
        "type": "connected",
        "model_type": p.model_type,
        "device": str(p.device),
        "message": "Connected to ASL Translation WebSocket server.",
        "mode": mode,
        "modes": ["sentence", "word"] if is_sign else ["sentence"],
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
                elif msg_type == "set_mode":
                    want = data.get("mode", "sentence")
                    if want in ("sentence", "word") and (is_sign or want == "sentence"):
                        mode = want
                        segmenter.pause_frames = WORD_PAUSE if mode == "word" else SENTENCE_PAUSE
                        segmenter.reset()
                    await websocket.send_json({"type": "mode_ack", "mode": mode})
                    continue
                elif msg_type == "reset":
                    segmenter.reset()
                    await websocket.send_json({"type": "reset_ack", "message": "Segmenter reset"})
                    continue
                elif msg_type == "force_cut":
                    # Force translation of currently buffered frames (fallback to preroll if needed)
                    frames_to_translate = (
                        list(segmenter.current_frames)
                        if len(segmenter.current_frames) >= 3
                        else list(segmenter.preroll_buffer)
                    )
                    if len(frames_to_translate) >= 3:
                        seg_arr = np.array(frames_to_translate, dtype=np.float32)
                        segmenter.reset()
                        await websocket.send_json({
                            "type": "translating",
                            "num_frames": len(seg_arr),
                            "reason": "force_cut",
                        })
                        t_res = await asyncio.to_thread(run_translate, seg_arr)
                        await websocket.send_json(result_msg(t_res, "force_cut"))
                    else:
                        await websocket.send_json({
                            "type": "notice",
                            "message": "No signing frames buffered yet — raise hands in view or simulate a clip first.",
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

            # Sign mode: stream live word candidates while the user is signing
            if is_sign and info["is_signing"] and frame_counter % 8 == 0 and len(segmenter.current_frames) >= 10:
                cur = np.array(segmenter.current_frames, dtype=np.float32)
                if mode == "word":
                    words = (await asyncio.to_thread(p.recognize_word, cur))[0]
                else:
                    words = await asyncio.to_thread(p.recognize, cur)
                await websocket.send_json({"type": "live_words", "words": [{"gloss": g, "conf": round(c, 2)} for g, c in words]})

            # Send periodic state update back to client (every 3 frames)
            if frame_counter % 3 == 0:
                await websocket.send_json({
                    "type": "status",
                    "is_signing": info["is_signing"],
                    "buffered_frames": info["buffered_frames"],
                    "pause_counter": info["pause_counter"],
                    "vel_norm": info["vel_norm"],
                    "lh_detected": info["lh_detected"],
                    "rh_detected": info["rh_detected"],
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
                t_res = await asyncio.to_thread(run_translate, segment_data)

                logger.info(
                    "Translated in %.1f ms: '%s'",
                    t_res["latency_ms"], t_res["text"]
                )

                # Push translation result to dashboard
                await websocket.send_json(result_msg(t_res, info["cut_reason"]))

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
    parser.add_argument("--model-type", default="sign", choices=["sign", "t5", "transformer", "lstm"], help="Model type")
    parser.add_argument("--config", default="configs/t5_decoder.yaml", help="Path to model config")
    parser.add_argument("--checkpoint", default="checkpoints/t5/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam width for translation")
    parser.add_argument("--min-frames", type=int, default=None, help="Min frames for sentence boundary")
    parser.add_argument("--pause-frames", type=int, default=None, help="Pause frames to cut sentence")
    parser.add_argument("--vocab", default=None, help="Word-list file restricting the sign vocabulary (default: data/aslcitizen/vocab_demo.txt; 'all' = every trained word)")
    parser.add_argument("--composer", default="auto",
                        choices=["auto", "qwen-1.5b-lora", "qwen-3b-lora", "qwen-3b", "qwen-1.5b", "none"],
                        help="Sentence model: auto = fine-tuned Qwen2.5-1.5B (best measured); qwen-3b = stock 3B "
                             "(needs ~8 GB free VRAM); none = show recognised words only")
    parser.add_argument("--min-raw", type=float, default=None,
                        help="Raw-probability floor for a sign (default 0.2). Higher = fewer false words for signs outside "
                             "the word list, lower = more words recognised. 0 = old behaviour")
    parser.add_argument("--ipcam-url", default=DEFAULT_IPCAM_URL, help="Default IP Webcam stream URL")
    args = parser.parse_args()

    server_config["model_type"] = args.model_type
    server_config["config"] = args.config
    server_config["checkpoint"] = args.checkpoint
    server_config["beam_size"] = args.beam_size
    server_config["min_frames"] = args.min_frames
    server_config["pause_frames"] = args.pause_frames
    server_config["ipcam_url"] = args.ipcam_url
    server_config["vocab"] = args.vocab
    server_config["min_raw"] = args.min_raw
    if args.composer != "auto":
        server_config["composer"] = None if args.composer == "none" else args.composer

    # Pre-load pipeline to verify checkpoint and warm up GPU before listening
    logger.info("Pre-warming ASLInferencePipeline...")
    _ = get_pipeline()
    logger.info("Pipeline ready. Starting Uvicorn on %s:%d...", args.host, args.port)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
