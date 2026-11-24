"""Score the gloss->English composer on the hand-written held-out set data/sentences/eval_gloss_en.tsv.
  coverage  : share of signed words that appear in the output (lemma match)          -> higher is better
  added     : share of sentences where the output invents a content word not signed -> lower is better
  qform     : question/statement form matches the reference (? vs .)                -> higher is better
  chrF      : character n-gram F-score vs the reference sentence                    -> higher is better
python scripts/eval_composer.py [--model Qwen/Qwen2.5-1.5B-Instruct] [--adapter DIR] [--device cuda|cpu] [--show N]"""
import sys, csv, re, argparse, time
sys.path.insert(0, '.')
import sacrebleu
from src.sign.faithful import score

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None); ap.add_argument("--device", default="cuda"); ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--file", default="data/sentences/eval_gloss_en.tsv")
    a = ap.parse_args()
    from src.sign.composer import GlossComposer
    comp = GlossComposer(a.model, a.device, adapter=a.adapter)   # adapter None = stock few-shot model
    rows = list(csv.DictReader(open(a.file), delimiter="\t"))
    hyps, refs, covs, adds, qf, bad = [], [], [], [], [], []; t0 = time.time()
    for r in rows:
        out = comp.compose(r["gloss"].split(), guard=False)["text"]   # score the raw model, not the safety fallback
        cov, added = score(r["gloss"], out)
        hyps.append(out); refs.append(r["english"]); covs.append(cov); adds.append(bool(added))
        qf.append(out.strip().endswith("?") == r["english"].strip().endswith("?"))
        if cov < 1 or added or not qf[-1]: bad.append((r["gloss"], out, r["english"], added))
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    print(f"model {comp.model_name}{' + '+a.adapter if a.adapter else ''} on {comp.device} | n={len(rows)} | {time.time()-t0:.0f}s")
    print(f"coverage {sum(covs)/len(covs):.3f} | added-content {sum(adds)/len(adds):.3f} | qform {sum(qf)/len(qf):.3f} | chrF {chrf:.1f}")
    for g, o, e, ad in bad[:a.show]: print(f"  {g.upper():32s} -> {o!r:45s} ref {e!r} {'+'+','.join(ad) if ad else ''}")

if __name__ == "__main__":
    main()
