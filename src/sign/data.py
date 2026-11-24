"""Dataset + layout-aware augmentation for isolated-sign landmarks.

Feature layout (static 600 + velocity 600):
  pose 0:68 (17x4) | LH 68:152 (21x4) | RH 152:236 (21x4) | face 236:472 (59x4)
  LH wrist-rel 472:535 (21x3) | RH wrist-rel 535:598 (21x3) | flags 598:600
"""
from __future__ import annotations
import csv, math, random, re
from pathlib import Path
import numpy as np, torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
LM_DIR = ROOT / "data/wlasl/landmarks"
G4 = np.arange(0, 472, 4)            # start of each 4-wide landmark
G3 = np.arange(472, 598, 3)          # start of each 3-wide wrist-relative landmark
HAND = {"L": (slice(68, 152), slice(472, 535), 598), "R": (slice(152, 236), slice(535, 598), 599)}
MAX_T = 96


CIT_DIR = ROOT / "data/aslcitizen/landmarks"
SOURCES = {"wlasl": ROOT / "data/wlasl/manifest.csv", "citizen": ROOT / "data/aslcitizen/manifest.csv"}


def lm_path(vid: str) -> Path:
    """'c:<id>' = ASL Citizen clip, anything else = WLASL/custom clip."""
    return CIT_DIR / f"{vid[2:]}.npy" if vid.startswith("c:") else LM_DIR / f"{vid}.npy"


def norm_gloss(g: str) -> str:
    """Merge naming variants: 'APPLE1' / 'apple' -> 'apple'; 'THANK-YOU' / 'thank you' -> 'thank_you'."""
    g = re.sub(r"\d+$", "", g.strip().lower())
    return re.sub(r"[\s\-]+", "_", g).strip("_")


def load_manifest(top_k: int, seed: int = 0, vocab: str | None = None, sources=("wlasl",), min_train: int = 0):
    """Official signer-disjoint splits of each source. Vocabulary = vocab file, else top_k by clip count."""
    rows = []
    for src in sources:
        if not SOURCES[src].exists():
            continue
        for r in csv.DictReader(open(SOURCES[src])):
            vid = r["video_id"] if src == "wlasl" else "c:" + r["video_id"]
            rows.append((vid, norm_gloss(r["gloss"]), r["split"]))
    cnt: dict[str, int] = {}; ntr: dict[str, int] = {}
    for _, g, sp in rows:
        cnt[g] = cnt.get(g, 0) + 1
        if sp == "train": ntr[g] = ntr.get(g, 0) + 1
    if min_train:
        cnt = {g: c for g, c in cnt.items() if ntr.get(g, 0) >= min_train}
    glosses = sorted(cnt, key=lambda g: (-cnt[g], g))[:top_k]
    if vocab:
        glosses = [norm_gloss(w) for w in open(vocab) if w.strip() and norm_gloss(w) in cnt]
        glosses = list(dict.fromkeys(glosses))
    gi = {g: i for i, g in enumerate(sorted(glosses))}
    split: dict[str, list] = {"train": [], "val": [], "test": []}
    for vid, g, sp in rows:
        if g in gi and sp in split:
            split[sp].append((vid, gi[g]))
    return sorted(gi, key=gi.get), split


def add_custom(split: dict, glosses: list, repeat: int = 4) -> int:
    """Add the user's own webcam recordings (data/wlasl/custom/<word>__NN.npy), oversampled."""
    import shutil
    gi = {g: i for i, g in enumerate(glosses)}; n = 0  # custom files named <word>__NN.npy
    for f in sorted((ROOT / "data/wlasl/custom").glob("*__*.npy")):
        w = norm_gloss(f.stem.split("__")[0])
        if w not in gi:
            continue
        vid = f"custom_{f.stem}"
        shutil.copy(f, LM_DIR / f"{vid}.npy")
        split["train"] += [(vid, gi[w])] * repeat; n += 1
    return n


def _rot_scale_shift(static: np.ndarray, ang: float, sc: float, d: np.ndarray) -> np.ndarray:
    c, s = math.cos(ang), math.sin(ang)
    out = static.copy()
    for st, shift in ((G4, True), (G3, False)):
        xi, yi, zi = st, st + 1, st + 2
        # only transform landmarks that are present (non-zero); missing stay exactly zero
        present = (np.abs(static[:, xi]) + np.abs(static[:, yi]) + np.abs(static[:, zi])) > 0
        x, y, z = static[:, xi] * sc, static[:, yi] * sc, static[:, zi] * sc
        nx, ny = x * c - y * s, x * s + y * c
        if shift:
            nx, ny, z = nx + d[0], ny + d[1], z + d[2]
        out[:, xi] = np.where(present, nx, 0.0); out[:, yi] = np.where(present, ny, 0.0)
        out[:, zi] = np.where(present, z, 0.0)
    return out


def with_velocity(static: np.ndarray) -> np.ndarray:
    vel = np.zeros_like(static); vel[1:] = static[1:] - static[:-1]
    return np.concatenate([static, vel], 1)


def augment(arr: np.ndarray) -> np.ndarray:
    return augment_static(arr[:, :600].astype(np.float32))


# pose subset order = MediaPipe 0..16; left/right partner pairs
_POSE_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


def mirror(st: np.ndarray) -> np.ndarray:
    """Left-right mirror = what a left-handed signer / mirrored camera produces:
    negate x (shoulder-centred coords) and swap every left/right block."""
    out = st.copy()
    out[:, G4] *= -1; out[:, G3] *= -1                      # x of all 4-wide and 3-wide points
    for a, b in _POSE_PAIRS:
        A, B = slice(4 * a, 4 * a + 4), slice(4 * b, 4 * b + 4)
        out[:, A], out[:, B] = out[:, B].copy(), out[:, A].copy()
    for (la, lb) in ((slice(68, 152), slice(152, 236)), (slice(472, 535), slice(535, 598)), (598, 599)):
        out[:, la], out[:, lb] = out[:, lb].copy(), out[:, la].copy()
    return out


MIRROR_P = 0.5


def augment_static(st: np.ndarray) -> np.ndarray:
    if random.random() < MIRROR_P:
        st = mirror(st)
    T = len(st)
    # random temporal crop (keep >=80%), speed 0.8-1.2
    keep = random.uniform(0.8, 1.0); L = max(4, int(T * keep)); s0 = random.randint(0, T - L)
    st = st[s0:s0 + L]
    newT = max(4, int(round(len(st) / random.uniform(0.8, 1.2))))
    idx = np.linspace(0, len(st) - 1, newT)
    lo = np.floor(idx).astype(int); hi = np.minimum(lo + 1, len(st) - 1); w = (idx - lo)[:, None]
    flags = st[:, 598:600]
    st = st[lo] * (1 - w) + st[hi] * w
    st[:, 598:600] = np.maximum(flags[lo], flags[hi])
    st = _rot_scale_shift(st, math.radians(random.uniform(-12, 12)), random.uniform(0.9, 1.1),
                          np.random.normal(0, 0.03, 3))
    st[:, :472] += np.where(st[:, :472] != 0, np.random.normal(0, 0.004, st[:, :472].shape), 0).astype(np.float32)
    # simulate tracker dropouts: zero a hand on random frames
    for side, (hs, rs, fi) in HAND.items():
        drop = np.random.rand(len(st)) < 0.06
        st[drop, hs] = 0; st[drop, rs] = 0; st[drop, fi] = 0
    return with_velocity(st)


def _zero_hands(st: np.ndarray) -> np.ndarray:
    st = st.copy()
    for hs, rs, fi in HAND.values():
        st[:, hs] = 0; st[:, rs] = 0; st[:, fi] = 0
    return st


def make_negative(pool: list, rng: random.Random) -> np.ndarray:
    """Synthetic 'no sign' sample (static 600-dim, velocity added by caller):
    rest posture (hands down/undetected) or a sign-to-sign transition window."""
    def clip():
        return np.load(lm_path(rng.choice(pool)[0])).astype(np.float32)[:, :600]
    kind = rng.random()
    if kind < 0.35:      # rest: first/last frames of a clip with hands removed
        a = clip(); n = rng.randint(8, 24)
        seg = a[:n] if rng.random() < 0.5 else a[-n:]
        return _zero_hands(seg)
    a, b = clip(), clip()
    ta = a[-max(3, int(len(a) * rng.uniform(0.1, 0.35))):]
    hb = b[:max(3, int(len(b) * rng.uniform(0.1, 0.35)))]
    gap = _zero_hands(a[-1:]).repeat(rng.randint(0, 8), 0)
    return np.concatenate([ta, gap, hb]) if len(gap) else np.concatenate([ta, hb])


class SignDataset(Dataset):
    """Items are (video_id, label). Label == num_real means the synthetic '_none_' class."""
    def __init__(self, items, train: bool, none_frac: float = 0.10, num_real: int | None = None):
        self.items, self.train, self.none_frac = list(items), train, none_frac
        self.K = num_real if num_real is not None else max(y for _, y in self.items) + 1
        self.pool = list(items)
        self.fixed_neg = 0 if train else int(len(self.items) * none_frac)   # deterministic negatives for val/test

    def __len__(self): return len(self.items) + self.fixed_neg

    def __getitem__(self, i):
        if i >= len(self.items):                                   # fixed negative (eval)
            return torch.from_numpy(with_velocity(make_negative(self.pool, random.Random(i)))), self.K
        if self.train and random.random() < self.none_frac:        # on-the-fly negative (train)
            neg = make_negative(self.pool, random)
            return torch.from_numpy(augment_static(neg)), self.K
        vid, y = self.items[i]
        a = np.load(lm_path(vid)).astype(np.float32)[:MAX_T]
        if self.train: a = augment(a)
        return torch.from_numpy(a[:MAX_T]), y


def collate(b):
    T = max(len(x) for x, _ in b)
    X = torch.zeros(len(b), T, 1200); M = torch.zeros(len(b), T, dtype=torch.bool)
    for i, (x, _) in enumerate(b):
        X[i, :len(x)] = x; M[i, :len(x)] = True
    return X, M, torch.tensor([y for _, y in b])
