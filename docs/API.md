# Server API

`python -m src.serve.server --port 8000` serves the dashboard at `/` and the endpoints below. Everything is
same-origin, so the dashboard works unchanged behind a tunnel. On an HTTPS page it switches to `wss://` automatically.

## HTTP endpoints

| Method and path | Returns |
|---|---|
| `GET /` | The dashboard (`app/index.html`) |
| `GET /health` | `{"status": "healthy", "model_type": "sign", "device": "cuda", "beam_size": 1}` |
| `GET /api/status` | Model type, device, checkpoint path, default IP-webcam URL |
| `GET /api/vocab` | `{"count": 366, "words": ["afternoon", "again", ...]}`: words currently recognisable |
| `GET /api/sample-clips` | Test clips the dashboard can simulate (`<word>__<video_id>.npy`) |
| `GET /api/clip/{split}/{clip_name}` | Landmark frames of one sample clip |
| `POST /api/translate` | Body `{"landmarks": [[1200 floats], ...]}`; same result object as a WebSocket `translation` |
| `GET /api/ipcam/test?url=...` | Checks that an Android IP Webcam stream is reachable |
| `GET /api/ipcam/stream?url=...` | Proxies the MJPEG stream (avoids browser CORS problems) |
| `GET /api/ipcam/snapshot?url=...` | One JPEG frame from the IP camera |

## WebSocket `/ws/translate`

### Client → server

| Message | Meaning |
|---|---|
| Binary, 4,800 bytes | One frame: 1,200 little-endian float32 values (layout in [ARCHITECTURE.md](ARCHITECTURE.md#1-landmarks-browser-and-python-identical)), 15 frames per second |
| `{"type": "frame", "landmarks": [1200 floats]}` | Same as above, as JSON |
| `{"type": "force_cut"}` | Translate whatever is buffered now (needs ≥ 3 frames, otherwise a `notice` is sent) |
| `{"type": "set_mode", "mode": "word"}` | `"word"`: one sign → that word, ~0.6 s pause ends it, no sentence composer. `"sentence"` (default): several signs → English sentence, ~1.4 s pause. Server replies `mode_ack` |
| `{"type": "reset"}` | Clear the segmenter buffer |
| `{"type": "ping"}` | Server answers `pong` |

Frames with the wrong size are ignored.

### Server → client

| `type` | When | Fields |
|---|---|---|
| `connected` | On connect | `model_type`, `device`, `message`, `mode`, `modes` |
| `mode_ack` | After `set_mode` | `mode`: the mode now active |
| `status` | Every 3 frames | `is_signing`, `buffered_frames`, `pause_counter`, `vel_norm`, `lh_detected`, `rh_detected` |
| `live_words` | Every 8 frames while signing (once ≥ 10 frames are buffered) | `words: [{"gloss", "conf"}]`: words recognised so far in this sentence |
| `translating` | Sentence finished, recognition starts | `num_frames`, `reason` (`pause_detected`, `max_duration_reached`, `force_cut`) |
| `translation` | Result | see below |
| `notice` | `force_cut` with nothing buffered | `message` |
| `pong` / `reset_ack` | Replies | `time` / `message` |

Example `translation`:

```json
{
  "type": "translation",
  "text": "I will go to school tomorrow.",
  "glosses": [{"gloss": "tomorrow", "conf": 0.91}, {"gloss": "me", "conf": 0.84},
              {"gloss": "go", "conf": 0.88}, {"gloss": "school", "conf": 0.79}],
  "alternatives": [],
  "mode": "sentence",
  "num_tokens": 4,
  "num_frames": 96,
  "latency_ms": 512.4,
  "fps_processed": 187.3,
  "reason": "pause_detected",
  "timestamp": 1764000000.0
}
```

- `text` is empty when no sign was recognised.
- When the faithfulness check rejects the composed sentence, `text` is the recognised words joined by commas.
- `tokens` is always an empty list in sign mode; it is kept for compatibility with the legacy T5 model.
- In word mode, `glosses` holds at most one word, `text` is that word ("Thank you."), and `alternatives` lists the
  three likeliest words with their probabilities (shown as "Other guesses" on the dashboard).

### Minimal Python client

```python
import asyncio, json, numpy as np, websockets

async def main(frames):                       # frames: (T, 1200) float32 landmark array
    async with websockets.connect("ws://localhost:8000/ws/translate") as ws:
        print(json.loads(await ws.recv())["type"])          # connected
        for f in frames:
            await ws.send(f.astype(np.float32).tobytes())
        await ws.send(json.dumps({"type": "force_cut"}))
        while (m := json.loads(await ws.recv()))["type"] != "translation":
            pass
        print(m["text"], [g["gloss"] for g in m["glosses"]])

asyncio.run(main(np.load("data/wlasl/landmarks/<clip>.npy")))
```
