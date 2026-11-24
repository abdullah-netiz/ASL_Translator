import json,sys,glob,os
rows=[]
for f in glob.glob('checkpoints/sign/sweep/*/results.json'):
    r=json.load(open(f)); rows.append((str(r['top_k']),r['best_val_loss'],os.path.basename(os.path.dirname(f)),r['best_epoch'],r['val_top1_at_best'],r['test_top1'],r['test_top5']))
for r in sorted(rows): print("K=%s valloss %.3f %-22s ep%-4d val1 %.3f | test1 %.3f test5 %.3f"%r)
