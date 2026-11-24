"""Chain held-out test clips (unseen signers) into 3-sign streams and check word recovery.
python scripts/e2e_test.py [n_streams] [vocab_file] [sources]"""
import sys, random, numpy as np
sys.path.insert(0, '.')
from src.sign.data import load_manifest, lm_path, with_velocity
from src.sign.recognizer import SignTranslator
n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
vocab = sys.argv[2] if len(sys.argv) > 2 else 'data/aslcitizen/vocab_demo.txt'
sources = tuple((sys.argv[3] if len(sys.argv) > 3 else 'wlasl,citizen').split(','))
words = [w.strip() for w in open(vocab) if w.strip()]
mglo, split = load_manifest(9999, vocab=vocab, sources=sources)     # label y indexes mglo
tr = SignTranslator(active_vocab=vocab)
random.seed(1); test = split['test']; hit = tot = exact = 0; shown = 0
for _ in range(n):
    clips = random.sample(test, 3); parts = []
    for v, y in clips:
        parts.append(np.load(lm_path(v)).astype(np.float32)[:, :600])
        parts.append(np.zeros((5, 600), np.float32))                  # short rest gap between signs
    seg = with_velocity(np.concatenate(parts)).astype(np.float32)
    truth = [mglo[y] for _, y in clips]; r = tr.translate(seg); got = [g['gloss'] for g in r['glosses']]
    hit += sum(1 for t in truth if t in got); tot += 3; exact += truth == got
    if shown < 8: print('TRUE', truth, '| GOT', got, '|', r['text']); shown += 1
print(f"word recall {hit/tot:.2f} | exact sequence {exact/n:.2f} over {n} streams ({len(test)} test clips, {sources})")
