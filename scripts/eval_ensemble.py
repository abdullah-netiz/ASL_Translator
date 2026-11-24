"""Evaluate an ensemble directory on the official test split. python scripts/eval_ensemble.py checkpoints/sign/ens_val"""
import sys, glob, torch, numpy as np
sys.path.insert(0, '.')
from torch.utils.data import DataLoader
from src.sign.data import SignDataset, collate, load_manifest
from src.sign.model import SignClassifier
d = sys.argv[1]; cks = [torch.load(f, map_location='cpu') for f in sorted(glob.glob(d + '/*/best.pt')) + sorted(glob.glob(d + '/*.pt'))]
glosses = cks[0]['glosses']; _, split = load_manifest(9999, vocab='data/wlasl/vocab_convo.txt')
ms = []
for c in cks:
    m = SignClassifier(**c['cfg']); m.load_state_dict(c['model']); ms.append(m.cuda().eval())
res = {}
for sp in ('val', 'test'):
    P, Y = [], []
    with torch.no_grad():
        for X, M, y in DataLoader(SignDataset(split[sp], False, none_frac=0.0, num_real=len(glosses) - 1), 128, collate_fn=collate):
            P.append(torch.stack([m(X.cuda(), M.cuda()).softmax(-1) for m in ms]).mean(0).cpu()); Y.append(y)
    P, Y = torch.cat(P), torch.cat(Y); t5 = P.topk(5, -1).indices
    res[sp] = ((P.argmax(-1) == Y).float().mean().item(), (t5 == Y[:, None]).any(1).float().mean().item(), P, Y)
    print(f"{d}: {len(ms)} models | {sp} top1 {res[sp][0]:.3f} top5 {res[sp][1]:.3f}")
P, Y = res['test'][2], res['test'][3]; pred = P.argmax(-1)
acc = {g: ((pred == i) & (Y == i)).sum().item() / max(1, (Y == i).sum().item()) for i, g in enumerate(glosses[:-1]) if (Y == i).any()}
print('best-recognised words:', ' '.join(f"{g}({a:.0%})" for g, a in sorted(acc.items(), key=lambda x: -x[1])[:25]))
print('weak words:', ' '.join(g for g, a in acc.items() if a == 0))
