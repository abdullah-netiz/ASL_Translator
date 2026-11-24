"""Evaluate checkpoints (one dir of */best.pt or *.pt = ensemble) on official test splits.
Reports full-vocabulary accuracy per source and accuracy when outputs are restricted to a word list.
  python scripts/eval_vocab.py <ckpt_dir> [--split test] [--lists a.txt,b.txt]"""
import argparse, glob, json, sys, torch
sys.path.insert(0, '.')
from torch.utils.data import DataLoader
from src.sign.data import SignDataset, collate, load_manifest, norm_gloss
from src.sign.model import SignClassifier
ap = argparse.ArgumentParser(); ap.add_argument("d"); ap.add_argument("--split", default="test")
ap.add_argument("--lists", default="data/aslcitizen/vocab_everyday.txt,data/wlasl/vocab_convo.txt")
ap.add_argument("--sources", default="wlasl,citizen"); a = ap.parse_args()
files = sorted(glob.glob(a.d + '/*/best.pt')) + sorted(glob.glob(a.d + '/*.pt')) or [a.d]
cks = [torch.load(f, map_location='cpu') for f in files]
glosses = cks[0]['glosses']; real = glosses[:-1]; gi = {g: i for i, g in enumerate(real)}
args = cks[0].get('args', {})
_, split = load_manifest(10**6, sources=tuple(a.sources.split(',')))
_g, _ = load_manifest(10**6, sources=tuple(a.sources.split(',')))
# remap manifest labels -> model label space
items = [(v, gi[_g[y]]) for v, y in split[a.split] if _g[y] in gi]
ms = []
for c in cks:
    m = SignClassifier(**c['cfg']); m.load_state_dict(c['model']); ms.append(m.cuda().eval())
P, Y, V = [], [], []
with torch.no_grad():
    for X, M, y in DataLoader(SignDataset(items, False, none_frac=0.0, num_real=len(real)), 256, collate_fn=collate, num_workers=2):
        P.append(torch.stack([m(X.cuda(), M.cuda()).softmax(-1) for m in ms]).mean(0)[:, :len(real)].cpu()); Y.append(y)
P, Y = torch.cat(P), torch.cat(Y); src = torch.tensor([v.startswith('c:') for v, _ in items])
def acc(p, y):
    return (p.argmax(-1) == y).float().mean().item(), (p.topk(min(5, p.shape[1]), -1).indices == y[:, None]).any(1).float().mean().item(), len(y)
res = {"models": len(ms), "vocab": len(real)}
for name, sel in (("all", torch.ones_like(src)), ("citizen", src), ("wlasl", ~src)):
    if sel.any(): res[f"full_{name}"] = acc(P[sel], Y[sel])
for lf in a.lists.split(','):
    words = [norm_gloss(w) for w in open(lf) if w.strip()]
    idx = torch.tensor(sorted({gi[w] for w in words if w in gi}))
    remap = {int(j): k for k, j in enumerate(idx)}
    sel = torch.tensor([int(y) in remap for y in Y])
    if sel.any():
        yy = torch.tensor([remap[int(y)] for y in Y[sel]])
        for name, s2 in (("citizen", src[sel]), ("wlasl", ~src[sel])):
            if s2.any(): res[f"{lf.split('/')[-1][:-4]}[{len(idx)}]_{name}"] = acc(P[sel][:, idx][s2], yy[s2])
for k, v in res.items():
    print(f"{k:34s}", v if not isinstance(v, tuple) else f"top1 {v[0]:.3f}  top5 {v[1]:.3f}  (n={v[2]})")
