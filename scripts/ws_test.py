"""Drive the real FastAPI /ws/translate with held-out test clips (unseen signers).
python scripts/ws_test.py [n_trials] [sentence|word]
  sentence: 3 clips chained per trial, checks the recognised word sequence
  word    : 1 clip per trial in word mode, checks the single returned word"""
import sys, random, json, numpy as np
sys.path.insert(0, '.')
from fastapi.testclient import TestClient
from src.serve import server
from src.sign.data import load_manifest, lm_path, with_velocity
n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
mode = sys.argv[2] if len(sys.argv) > 2 else "sentence"
k = 3 if mode == "sentence" else 1
server.server_config.update({"model_type": "sign", **({"min_raw": float(sys.argv[3])} if len(sys.argv) > 3 else {})})
mglo, split = load_manifest(9999, vocab=str(server.DEMO_VOCAB), sources=("wlasl", "citizen")); random.seed(7)
c = TestClient(server.app); hit = tot = exact = 0
with c.websocket_connect("/ws/translate") as ws:
    print(ws.receive_json()["type"])
    ws.send_text(json.dumps({"type": "set_mode", "mode": mode}))
    while (m := ws.receive_json())["type"] != "mode_ack": pass
    print("mode", m["mode"])
    for trial in range(n):
        clips = random.sample(split['test'], k); parts = []
        for v, y in clips:
            parts += [np.load(lm_path(v)).astype(np.float32)[:, :600], np.zeros((4, 600), np.float32)]
        seg = with_velocity(np.concatenate(parts)).astype(np.float32)
        truth = [mglo[y] for _, y in clips]
        for fr in np.concatenate([seg, np.zeros((30, 1200), np.float32)]):
            ws.send_bytes(fr.astype(np.float32).tobytes())
        ws.send_text(json.dumps({"type": "ping"}))
        msgs = []                                   # a stream may be cut into several segments: keep all
        while True:
            m = ws.receive_json()
            if m["type"] == "translation": msgs.append(m)
            if m["type"] == "pong": break
        got = [g["gloss"] for m in msgs for g in m["glosses"]]
        hit += sum(t in got for t in truth); tot += k; exact += got == truth
        alts = [a["gloss"] for m in msgs for a in m.get("alternatives", [])]
        print("TRUE", truth, "->", got, "|", " / ".join(m["text"] for m in msgs), f"({len(msgs)} seg)", f"alts {alts}" if mode == "word" else "")
print(f"[{mode}] word recall {hit/tot:.2f} | exact {exact/n:.2f} over {n} trials")
