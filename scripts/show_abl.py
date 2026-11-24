import json,glob,os,collections,sys
d=sys.argv[1] if len(sys.argv)>1 else 'checkpoints/sign/abl'
g=collections.defaultdict(list)
for f in glob.glob(d+'/*/results.json'):
    r=json.load(open(f)); g[os.path.basename(os.path.dirname(f)).rsplit('_',1)[0]].append(r)
for k,v in sorted(g.items(), key=lambda kv: sum(r['best_val_loss'] for r in kv[1])/len(kv[1])):
    m=lambda key: sum(r[key] for r in v)/len(v)
    print(f"{k:12s} n={len(v)} val_loss {m('best_val_loss'):.3f} | val1 {m('val_top1_at_best'):.3f} | test1 {m('test_top1'):.3f} test5 {m('test_top5'):.3f} | ep {m('best_epoch'):.0f}")
