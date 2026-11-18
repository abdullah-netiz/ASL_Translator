"""
scripts/test_sign_e2e.py — Concrete End-to-End Test Cases for ASL Translation.

Runs 4 diagnostic test cases to verify sign detection, segmentation, translation,
WebSocket streaming, and (optionally) live IP Webcam landmark detection:
  1. Synthetic signing -> rest pause segmentation test
  2. Real How2Sign .mp4 video -> MediaPipe Holistic -> Normalize -> Segmenter -> T5
  3. Live WebSocket server test (ws://127.0.0.1:8000/ws/translate)
  4. Optional IP Webcam snapshot -> MediaPipe Holistic hand/pose detection check

Usage:
  .venv/bin/python scripts/test_sign_e2e.py
  .venv/bin/python scripts/test_sign_e2e.py --ipcam-url http://172.15.45.93:8080/video
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import httpx
import numpy as np
import yaml

from src.extract.holistic import HolisticExtractor
from src.extract.normalize import normalize_sequence
from src.serve.inference import ASLInferencePipeline
from src.serve.segmenter import ContinuousSignSegmenter


def load_val_manifest() -> list[dict[str, str]]:
    manifest = _REPO_ROOT / "data" / "manifests" / "val_manifest.csv"
    if not manifest.exists():
        return []
    with open(manifest, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_case_1_segmenter() -> None:
    print("\n" + "=" * 65)
    print("TEST CASE 1: Synthetic Signing + Pause Boundary Detection")
    print("=" * 65)
    seg = ContinuousSignSegmenter(min_frames=12, pause_frames=12, motion_threshold=0.08)

    # 25 active signing frames (hands raised, moving)
    active = np.zeros(1200, dtype=np.float32)
    active[598] = 1.0  # Left hand detected
    active[599] = 1.0  # Right hand detected
    active[61] = 0.5   # Left wrist elevated
    active[63] = 0.9   # Left wrist visibility
    active[65] = 0.5   # Right wrist elevated
    active[67] = 0.9   # Right wrist visibility
    active[600:650] = 0.25  # Gentle signing motion

    # 15 rest frames (hands dropped)
    rest = np.zeros(1200, dtype=np.float32)

    emitted = []
    for i in range(25):
        ready, data, info = seg.process_frame(active)
        if ready:
            emitted.append((i, data, info))
    for i in range(15):
        ready, data, info = seg.process_frame(rest)
        if ready:
            emitted.append((25 + i, data, info))

    assert len(emitted) == 1, f"Expected 1 segment, got {len(emitted)}"
    frame_idx, seg_arr, info = emitted[0]
    print(f"PASS: Cut sentence at frame {frame_idx} | length={len(seg_arr)} frames | reason={info['cut_reason']}")


def test_case_2_real_video_pipeline() -> None:
    print("\n" + "=" * 65)
    print("TEST CASE 2: Real How2Sign .mp4 Video -> MediaPipe -> Segmenter -> T5")
    print("=" * 65)
    rows = load_val_manifest()
    # Pick a clean validation video that exists on disk (e.g. -d5dN54tH2E_2-1-rgb_front)
    target_row = None
    for preferred in ("-d5dN54tH2E_2-1-rgb_front", "-d5dN54tH2E_0-1-rgb_front", "-d5dN54tH2E_10-1-rgb_front"):
        for r in rows:
            if r["clip_id"] == preferred and Path(r["video_path"]).exists():
                target_row = r
                break
        if target_row:
            break
    if target_row is None:
        for r in rows:
            if Path(r["video_path"]).exists():
                target_row = r
                break

    assert target_row is not None, "No validation .mp4 videos found on disk!"
    video_path = Path(target_row["video_path"])
    print(f"Video     : {video_path.name}")
    print(f"Ground GT : {target_row['sentence']}")

    with open(_REPO_ROOT / "configs" / "landmarks.yaml") as f:
        lm_cfg = yaml.safe_load(f)

    # 1. Extract landmarks from raw MP4 using MediaPipe Holistic (stride=2 -> 15 fps)
    extractor = HolisticExtractor(lm_cfg)
    raw_frames = extractor.extract_video(video_path, stride=2)
    extractor.close()

    norm_seq = normalize_sequence(raw_frames, lm_cfg)
    lh_rate = float(norm_seq[:, 598].mean()) * 100.0
    rh_rate = float(norm_seq[:, 599].mean()) * 100.0
    vel_mean = float(np.linalg.norm(norm_seq[:, 600:], axis=1).mean())
    print(f"Extracted : {len(norm_seq)} frames @ 15fps | LH={lh_rate:.1f}% | RH={rh_rate:.1f}% | mean_vel={vel_mean:.2f}")

    # 2. Stream frames + 15 rest frames into ContinuousSignSegmenter
    seg = ContinuousSignSegmenter()
    segments = []
    for f in norm_seq:
        ready, data, _ = seg.process_frame(f.astype(np.float32))
        if ready and data is not None:
            segments.append(data)
    rest = np.zeros(1200, dtype=np.float32)
    for _ in range(15):
        ready, data, _ = seg.process_frame(rest)
        if ready and data is not None:
            segments.append(data)

    assert len(segments) >= 1, "Segmenter did not detect any sentence segment from the real video!"
    print(f"Segmented : {len(segments)} sentence(s) detected (first segment = {len(segments[0])} frames)")

    # 3. Translate with T5
    pipeline = ASLInferencePipeline("t5")
    res = pipeline.translate(segments[0], beam_size=5, max_length=48)
    print(f"Predicted : {res['text']!r}")
    print(f"Latency   : {res['latency_ms']} ms (target < 2000 ms)")
    assert res["text"].strip() != "", "Model returned empty text!"
    print("PASS: Real video end-to-end pipeline succeeded.")


async def test_case_3_websocket_server(server_url: str) -> None:
    print("\n" + "=" * 65)
    print(f"TEST CASE 3: Live WebSocket Server ({server_url})")
    print("=" * 65)
    try:
        import websockets
    except ImportError:
        print("SKIP: websockets package not installed.")
        return

    http_base = server_url.replace("ws://", "http://").replace("wss://", "https://").rstrip("/")
    try:
        health = httpx.get(f"{http_base}/health", timeout=3.0)
        health.raise_for_status()
        print(f"Server OK : {health.json()}")
    except Exception as exc:
        print(f"SKIP: Server not reachable at {http_base} ({exc}). Start it with `make serve`.")
        return

    clip_name = "-d5dN54tH2E_2-1-rgb_front.npy"
    clip_resp = httpx.get(f"{http_base}/api/clip/val/{clip_name}", timeout=5.0)
    clip_resp.raise_for_status()
    clip = clip_resp.json()
    print(f"Clip      : {clip_name} ({clip['num_frames']} frames)")
    print(f"Ground GT : {clip.get('reference_text', '')}")

    ws_endpoint = f"{server_url.rstrip('/')}/ws/translate"
    translations = []
    async with websockets.connect(ws_endpoint) as ws:
        _ = await ws.recv()  # connected banner
        for frame in clip["frames"]:
            await ws.send(np.array(frame, dtype=np.float32).tobytes())
        rest = np.zeros(1200, dtype=np.float32).tobytes()
        for _ in range(15):
            await ws.send(rest)

        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=4.0)
                msg = json.loads(raw)
                if msg.get("type") == "translation":
                    translations.append(msg)
        except asyncio.TimeoutError:
            pass

    assert len(translations) >= 1, "No translation received over WebSocket!"
    for idx, t in enumerate(translations, 1):
        print(f"WS Result #{idx}: {t['text']!r} ({t['latency_ms']} ms, {t['num_frames']} frames, reason={t.get('reason')})")
    print("PASS: Live WebSocket streaming & translation succeeded.")


def test_case_4_ipcam_frame(ipcam_url: str) -> None:
    print("\n" + "=" * 65)
    print(f"TEST CASE 4: Live IP Webcam Snapshot & Hand Detection ({ipcam_url})")
    print("=" * 65)
    shot_url = ipcam_url.replace("/video", "/shot.jpg")
    try:
        resp = httpx.get(shot_url, timeout=4.0)
        resp.raise_for_status()
    except Exception as exc:
        print(f"FAIL: Cannot reach IP Webcam at {shot_url}: {exc}")
        print("  -> Check that the IP Webcam app is running on your phone and on the same Wi-Fi/subnet.")
        return

    img_arr = np.frombuffer(resp.content, dtype=np.uint8)
    frame_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        print("FAIL: Received bytes from IP Webcam could not be decoded as JPEG.")
        return

    h, w = frame_bgr.shape[:2]
    print(f"Captured  : {w}x{h} frame from {shot_url}")

    with open(_REPO_ROOT / "configs" / "landmarks.yaml") as f:
        lm_cfg = yaml.safe_load(f)

    extractor = HolisticExtractor(lm_cfg)
    raw = extractor.process_frame(frame_bgr)
    extractor.close()

    pose_ok = raw["pose"] is not None
    lh_ok = raw["left_hand"] is not None
    rh_ok = raw["right_hand"] is not None
    face_ok = raw["face"] is not None
    print(f"Detections: Pose={pose_ok} | LeftHand={lh_ok} | RightHand={rh_ok} | Face={face_ok}")
    if not lh_ok and not rh_ok:
        print("NOTE: Frame captured cleanly, but no hands were visible in this snapshot. Stand ~1m back with upper body & hands in frame.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E2E ASL Sign Translation Test Cases")
    parser.add_argument("--server-url", default="ws://127.0.0.1:8000", help="Running server base WS URL")
    parser.add_argument("--ipcam-url", default="", help="Optional IP Webcam URL to test live snapshot")
    args = parser.parse_args()

    test_case_1_segmenter()
    test_case_2_real_video_pipeline()
    asyncio.run(test_case_3_websocket_server(args.server_url))
    if args.ipcam_url:
        test_case_4_ipcam_frame(args.ipcam_url)


if __name__ == "__main__":
    main()
