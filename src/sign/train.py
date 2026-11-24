"""Train the isolated-sign classifier.  python -m src.sign.train --top-k 100"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from src.sign.data import SignDataset, collate, load_manifest, add_custom, ROOT
from src.sign.model import SignClassifier


def evaluate(model, dl, dev):
    model.eval(); c1 = c5 = n = 0; tl = 0.0
    with torch.no_grad():
        for X, M, y in dl:
            lg = model(X.to(dev), M.to(dev)); top = lg.topk(5, -1).indices.cpu()
            tl += nn.functional.cross_entropy(lg, y.to(dev), reduction="sum").item()
            c1 += (top[:, 0] == y).sum().item(); c5 += (top == y[:, None]).any(1).sum().item(); n += len(y)
    return c1 / n, c5 / n, tl / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=100); ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--out", default=None); ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--d-model", type=int, default=256); ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.2); ap.add_argument("--feat", default="all")
    ap.add_argument("--wd", type=float, default=0.05); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smooth", type=float, default=0.1); ap.add_argument("--vocab", default=None); ap.add_argument("--trainval", action="store_true"); ap.add_argument("--custom", action="store_true")
    ap.add_argument("--derived", action="store_true"); ap.add_argument("--mirror", type=float, default=0.0)
    ap.add_argument("--sources", default="wlasl", help="comma list: wlasl,citizen"); ap.add_argument("--min-train", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out or ROOT / f"checkpoints/sign/wlasl{a.top_k}"); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed); import random; random.seed(a.seed)
    import src.sign.data as D; D.MIRROR_P = a.mirror
    glosses, split = load_manifest(a.top_k, vocab=a.vocab, sources=tuple(a.sources.split(',')), min_train=a.min_train)
    if a.custom:
        print('custom clips added:', add_custom(split, glosses), flush=True)
    if a.trainval:   # deployment mode: fixed epochs, no early stopping, test set stays untouched
        split["train"] = split["train"] + split["val"]; a.patience = 10**9
    print({k: len(v) for k, v in split.items()}, "classes", len(glosses), flush=True)
    dev = torch.device("cuda")
    tr = DataLoader(SignDataset(split["train"], True, num_real=len(glosses)), a.bs, shuffle=True, num_workers=__import__('src.sign.resources', fromlist=['x']).safe_workers(0.6, cap=4), collate_fn=collate, drop_last=True, persistent_workers=True)
    va = DataLoader(SignDataset(split["val"], False, num_real=len(glosses)), 128, collate_fn=collate)
    te = DataLoader(SignDataset(split["test"], False, num_real=len(glosses)), 128, collate_fn=collate)
    model = SignClassifier(len(glosses) + 1, d_model=a.d_model, nhead=max(1, a.d_model // 32), layers=a.layers, dropout=a.dropout, feat=a.feat, derived=a.derived).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    total = a.epochs * len(tr); warm = 5 * len(tr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))
    crit = nn.CrossEntropyLoss(label_smoothing=a.smooth)
    best = 0; best_loss = float('inf'); bad = 0; best_ep = 0; t0 = time.time()
    for ep in range(a.epochs):
        model.train(); tl = 0
        for X, M, y in tr:
            X, M, y = X.to(dev), M.to(dev), y.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16): loss = crit(model(X, M), y)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); tl += loss.item()
        v1, v5, vl = evaluate(model, va, dev)
        best = max(best, v1)
        if a.trainval and ep + 1 == a.epochs:
            torch.save({"model": model.state_dict(), "cfg": model.cfg, "glosses": glosses + ["_none_"]}, out / "best.pt")
        if not a.trainval and vl < best_loss:   # freeze the weights with the LOWEST validation loss
            best_loss, best_ep, bad = vl, ep + 1, 0
            torch.save({"model": model.state_dict(), "cfg": model.cfg, "glosses": glosses + ["_none_"]}, out / "best.pt")
        else:
            bad += 1
        if ep % 5 == 4 or ep == 0:
            print(f"ep{ep+1} train {tl/len(tr):.3f} val_loss {vl:.3f} top1 {v1:.3f} top5 {v5:.3f} | best_loss {best_loss:.3f}@{best_ep} [{time.time()-t0:.0f}s]", flush=True)
        if bad >= a.patience:
            print(f"early stop at epoch {ep+1}: val loss rising since epoch {best_ep}", flush=True); break
    ck = torch.load(out / "best.pt"); model.load_state_dict(ck["model"])
    v1b, _, _ = evaluate(model, va, dev); t1, t5, tloss = evaluate(model, te, dev)
    res = {"args": vars(a), "top_k": a.top_k, "best_epoch": best_ep, "best_val_loss": best_loss, "val_top1_at_best": v1b, "test_loss": tloss, "test_top1": t1, "test_top5": t5, "n_train": len(split["train"])}
    json.dump(res, open(out / "results.json", "w"), indent=1); print("FINAL", res)


if __name__ == "__main__":
    main()
