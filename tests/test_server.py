"""
tests/test_server.py — Integration and Unit Tests for Server, Segmenter, and Inference.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch
from starlette.testclient import TestClient

from src.serve.inference import ASLInferencePipeline
from src.serve.segmenter import ContinuousSignSegmenter
from src.serve.server import app, server_config


@pytest.fixture
def segmenter():
    return ContinuousSignSegmenter(
        min_frames=10,
        max_frames=100,
        pause_frames=5,
        motion_threshold=0.5,
        wrist_drop_threshold=1.8,
    )


def test_segmenter_idle_on_empty_frames(segmenter):
    """Verifies that empty frames do not trigger signing."""
    empty_frame = np.zeros(1200, dtype=np.float32)
    ready, data, info = segmenter.process_frame(empty_frame)
    assert not ready
    assert data is None
    assert not info["is_signing"]


def test_segmenter_detects_active_signing_and_pause(segmenter):
    """Verifies that active frames transition to SIGNING and pause frames trigger segment cut."""
    # Create active signing frame: hands detected (indices 598, 599 = 1.0),
    # wrists raised (indices 61, 65 = 0.5), non-zero velocity
    active_frame = np.zeros(1200, dtype=np.float32)
    active_frame[598] = 1.0
    active_frame[599] = 1.0
    active_frame[61] = 0.5
    active_frame[65] = 0.5
    active_frame[600:] = 1.0  # Motion

    # Feed 15 active frames
    for _ in range(15):
        ready, data, info = segmenter.process_frame(active_frame)
        assert not ready
        assert info["is_signing"]

    # Now feed pause frames (empty / dropped hands)
    pause_frame = np.zeros(1200, dtype=np.float32)
    segment_emitted = False
    emitted_data = None

    for _ in range(6):
        ready, data, info = segmenter.process_frame(pause_frame)
        if ready:
            segment_emitted = True
            emitted_data = data
            break

    assert segment_emitted, "Sentence boundary was not triggered by pause"
    assert emitted_data is not None
    assert emitted_data.shape[1] == 1200
    assert emitted_data.shape[0] >= 10


def test_inference_pipeline_execution():
    """Verifies inference pipeline runs and produces text on synthetic landmarks."""
    pipeline = ASLInferencePipeline(
        model_type="t5",
        config_path="configs/t5_decoder.yaml",
        checkpoint_path="checkpoints/t5/best_model.pt",
    )
    dummy_landmarks = np.random.randn(30, 1200).astype(np.float32)
    result = pipeline.translate(dummy_landmarks, beam_size=1, max_length=20)

    assert "text" in result
    assert isinstance(result["text"], str)
    assert "latency_ms" in result
    assert result["num_frames"] == 30
    assert result["latency_ms"] < 2000.0, f"Latency {result['latency_ms']} ms exceeds 2s limit"


def test_fastapi_endpoints():
    """Verifies HTTP health and status endpoints."""
    server_config["model_type"] = "t5"
    server_config["config"] = "configs/t5_decoder.yaml"
    server_config["checkpoint"] = "checkpoints/t5/best_model.pt"

    client = TestClient(app)

    # Health check
    res_health = client.get("/health")
    assert res_health.status_code == 200
    data = res_health.json()
    assert data["status"] == "healthy"
    assert data["model_type"] == "t5"

    # Status check
    res_status = client.get("/api/status")
    assert res_status.status_code == 200

    # Sample clips
    res_clips = client.get("/api/sample-clips")
    assert res_clips.status_code == 200
    clips_data = res_clips.json()
    assert "val_clips" in clips_data


def test_websocket_stream_flow():
    """Verifies WebSocket endpoint receives frames and acknowledges."""
    server_config["model_type"] = "t5"
    server_config["config"] = "configs/t5_decoder.yaml"
    server_config["checkpoint"] = "checkpoints/t5/best_model.pt"

    client = TestClient(app)

    with client.websocket_connect("/ws/translate") as websocket:
        # Connection banner
        data = websocket.receive_json()
        assert data["type"] == "connected"

        # Ping pong
        websocket.send_json({"type": "ping"})
        pong = websocket.receive_json()
        assert pong["type"] == "pong"

        # Send empty frame (JSON)
        dummy_frame = [0.0] * 1200
        websocket.send_json({"type": "frame", "landmarks": dummy_frame})

        # Send reset
        websocket.send_json({"type": "reset"})
        reset_ack = websocket.receive_json()
        assert reset_ack["type"] == "reset_ack"
