"""Build a self-contained bundle to run the translator on a cloud GPU notebook (Kaggle / Colab / any Linux VM).

  python scripts/package_for_cloud.py               -> dist/asl_cloud_bundle.tar.gz   (code + models + word lists + demo clips)
  python scripts/package_for_cloud.py --with-tests  -> also the held-out test clips for ws_test / e2e_test / eval_vocab

The bundle does NOT contain Qwen itself (downloaded from Hugging Face on first start) or the raw datasets."""
import argparse, csv, sys, tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.sign.data import load_manifest, lm_path

ap = argparse.ArgumentParser(); ap.add_argument("--with-tests", action="store_true")
ap.add_argument("--out", default="dist/asl_cloud_bundle.tar.gz")
a = ap.parse_args()

files: list[Path] = []
def add(p):
    p = ROOT / p
    if p.is_dir():
        files.extend(f for f in p.rglob("*") if f.is_file() and "__pycache__" not in f.parts)
    elif p.exists():
        files.append(p)
    else:
        print("missing (skipped):", p.relative_to(ROOT))

# code + web app + configs used by imports
for p in ["src", "app", "configs", "scripts", "tests", "docs", "requirements.txt", "requirements-serve.txt",
          "pyproject.toml", "README.md", "reports/restart_report.md"]:
    add(p)
# models: sign ensemble (+ personal fine-tune if any) and composer adapters
for p in ["checkpoints/sign/final", "checkpoints/sign/final_custom", "checkpoints/composer/lora_1.5b", "checkpoints/composer/lora_3b"]:
    if (ROOT / p).exists(): add(p)
# word lists, manifests, sentence data
for p in ["data/aslcitizen/vocab_demo.txt", "data/aslcitizen/vocab_everyday.txt", "data/aslcitizen/all_words.txt",
          "data/wlasl/vocab_convo.txt", "data/wlasl/manifest.csv", "data/aslcitizen/manifest.csv", "data/sentences"]:
    add(p)
# demo clips for the dashboard's "Simulate" (WLASL test clips of the conversational words)
_, split = load_manifest(9999, vocab=str(ROOT / "data/wlasl/vocab_convo.txt"))
clips = {lm_path(v) for v, _ in split["test"]}
if a.with_tests:   # held-out test clips of the demo vocabulary, both datasets
    _, split = load_manifest(9999, vocab=str(ROOT / "data/aslcitizen/vocab_demo.txt"), sources=("wlasl", "citizen"))
    clips |= {lm_path(v) for v, _ in split["test"]}
files.extend(sorted(c for c in clips if c.exists()))

out = ROOT / a.out; out.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(out, "w:gz") as t:
    for f in sorted(set(files)):
        t.add(f, arcname=str(Path("ASL_Translator") / f.relative_to(ROOT)))
mb = out.stat().st_size / 1e6
print(f"{len(set(files))} files ({len(clips)} landmark clips) -> {out.relative_to(ROOT)} ({mb:.0f} MB)")
