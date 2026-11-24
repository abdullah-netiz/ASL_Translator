"""Live sign -> English: windowed sign classifier + Qwen gloss composer.

Drop-in replacement for ASLInferencePipeline (same .translate() result keys)."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np, torch
from src.sign.model import SignClassifier

ROOT = Path(__file__).resolve().parents[2]


class SignTranslator:
    model_type = "sign"

    def __init__(self, checkpoint: str | Path | None = None, composer_model: str | None = "Qwen/Qwen2.5-3B-Instruct",
                 device: str | None = None, scales=(20, 30, 42), stride: int = 3, min_conf: float = 0.5, overlap: float = 0.2,
                 active_vocab: str | Path | None = None, renorm: bool = False, min_raw: float = 0.2):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        path = Path(checkpoint or ROOT / "checkpoints/sign/final")
        files = sorted(path.glob("*.pt")) if path.is_dir() else [path]   # a directory = ensemble
        self.models = []
        for f in files:
            ck = torch.load(f, map_location="cpu")
            self.glosses = ck["glosses"]
            m = SignClassifier(**ck["cfg"]); m.load_state_dict(ck["model"])
            self.models.append(m.to(self.device).eval())
        self.model = self.models[0]
        # Optional active vocabulary: only these words (+ _none_) can be output.
        self.active = None
        if active_vocab and Path(active_vocab).exists():
            from src.sign.data import norm_gloss
            want = {norm_gloss(w) for w in open(active_vocab) if w.strip()}
            mask = np.array([g in want or g == "_none_" for g in self.glosses])
            if mask.sum() > 1:
                self.active = mask
        self.active_words = [g for i, g in enumerate(self.glosses) if g != "_none_" and (self.active is None or self.active[i])]
        # min_raw > 0 (default): a word needs >= min_conf share among listed words AND >= min_raw raw probability.
        # min_raw = 0 restores the old behaviour (renormalise over the listed words, min_conf only).
        renorm = renorm or min_raw == 0
        self.scales, self.stride, self.min_conf, self.overlap, self.renorm, self.min_raw = scales, stride, min_conf, overlap, renorm, min_raw
        self.beam_size = 1
        self.checkpoint_path = path
        self.composer = None
        if composer_model:
            from src.sign.composer import GlossComposer
            self.composer = GlossComposer(composer_model, str(self.device))
            self.composer_name = f"{self.composer.model_name} ({self.composer.device})"

    @torch.no_grad()
    def _windows(self, seg: np.ndarray):
        """Multi-scale sliding windows -> (start, end, probs[K]) incl. the _none_ class."""
        T = len(seg); spans = []
        for w in self.scales:
            if T <= w:
                spans.append((0, T))
            else:
                spans += [(s, s + w) for s in range(0, T - w + 1, self.stride)]
                if (T - w) % self.stride: spans.append((T - w, T))
        spans = sorted(set(spans))
        L = max(e - s for s, e in spans)
        X = torch.zeros(len(spans), L, 1200); M = torch.zeros(len(spans), L, dtype=torch.bool)
        for i, (s, e) in enumerate(spans):
            X[i, :e - s] = torch.from_numpy(seg[s:e]); M[i, :e - s] = True
        Xd, Md = X.to(self.device), M.to(self.device)
        p = torch.stack([m(Xd, Md).softmax(-1) for m in self.models]).mean(0).cpu().numpy()
        if self.active is not None:
            # Zero the inactive words but do NOT renormalise: a sign outside the list keeps its probability on the
            # (inactive) word it really is, so it is rejected instead of being forced onto the nearest listed word.
            p = p * self.active
            if self.renorm:
                p = p / p.sum(-1, keepdims=True).clip(1e-9)
        return [(s, e, p[i]) for i, (s, e) in enumerate(spans)]

    def recognize(self, seg: np.ndarray) -> list[tuple[str, float]]:
        """Pick confident, non-overlapping word spans (greedy NMS by confidence), return in time order."""
        none_idx = len(self.glosses) - 1
        cands = []
        for s, e, p in self._windows(seg):
            k = int(p[:none_idx].argmax()); raw = float(p[k])
            c = raw / max(float(p.sum()), 1e-9) if self.min_raw > 0 else raw   # hybrid: listed-word share + raw floor
            if p.argmax() != none_idx and c >= self.min_conf and raw >= self.min_raw and e - s >= 8:
                cands.append((c, s, e, k))
        cands.sort(reverse=True); chosen = []
        for c, s, e, k in cands:
            ok = True
            for c2, s2, e2, k2 in chosen:
                inter = max(0, min(e, e2) - max(s, s2))
                if inter / min(e - s, e2 - s2) > self.overlap: ok = False; break
            if ok: chosen.append((c, s, e, k))
        chosen.sort(key=lambda x: (x[1] + x[2]) / 2)
        out = []
        for c, s, e, k in chosen:       # collapse identical neighbours (same sign seen twice)
            if out and out[-1][0] == self.glosses[k] and s - out[-1][2] < 4: continue
            out.append((self.glosses[k], c, e))
        return [(g, c) for g, c, _ in out]

    def recognize_word(self, seg: np.ndarray, top: int = 3) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        """Word mode: the segment is ONE sign. Returns ([(best_word, conf)] or [], up to `top` alternatives).
        Same acceptance rule as recognize(); the best window over all scales wins, nothing is split."""
        none_idx = len(self.glosses) - 1
        best: dict[int, float] = {}; accepted = None
        for s, e, p in self._windows(seg):
            if e - s < 8 and len(seg) >= 8:
                continue
            k = int(p[:none_idx].argmax()); raw = float(p[k])
            c = raw / max(float(p.sum()), 1e-9) if self.min_raw > 0 else raw
            for j in np.argsort(p[:none_idx])[::-1][:top]:
                best[int(j)] = max(best.get(int(j), 0.0), float(p[j]))
            if p.argmax() != none_idx and c >= self.min_conf and raw >= self.min_raw:
                if accepted is None or c > accepted[1]:
                    accepted = (k, c)
        alts = sorted(best.items(), key=lambda kv: -kv[1])[:top]
        alts = [(self.glosses[j], round(v, 3)) for j, v in alts if v > 0]
        return ([(self.glosses[accepted[0]], accepted[1])] if accepted else []), alts

    @staticmethod
    def word_text(word: str) -> str:
        return word.replace("_", " ").capitalize() + "." if word else ""

    def translate_word(self, landmarks) -> dict:
        """Word mode result: same keys as translate(), text = the single recognised word (no composer)."""
        seg = np.asarray(landmarks, dtype=np.float32)
        t0 = time.perf_counter()
        words, alts = self.recognize_word(seg)
        ms = (time.perf_counter() - t0) * 1000
        return {"text": self.word_text(words[0][0]) if words else "", "glosses": [{"gloss": w, "conf": round(c, 3)} for w, c in words],
                "alternatives": [{"gloss": w, "conf": c} for w, c in alts], "mode": "word",
                "tokens": [], "num_tokens": len(words), "num_frames": len(seg),
                "latency_ms": round(ms, 2), "fps_processed": round(len(seg) / (ms / 1000), 1) if ms else 0}

    def translate(self, landmarks, beam_size=None, max_length=None) -> dict:
        seg = np.asarray(landmarks, dtype=np.float32)
        t0 = time.perf_counter()
        words = self.recognize(seg)
        text = ""
        if words:
            if self.composer:
                text = self.composer.compose([w for w, _ in words])["text"]
            else:
                text = " ".join(w for w, _ in words)
        ms = (time.perf_counter() - t0) * 1000
        return {"text": text, "glosses": [{"gloss": w, "conf": round(c, 3)} for w, c in words],
                "tokens": [], "num_tokens": len(words), "num_frames": len(seg),
                "latency_ms": round(ms, 2), "fps_processed": round(len(seg) / (ms / 1000), 1) if ms else 0}
