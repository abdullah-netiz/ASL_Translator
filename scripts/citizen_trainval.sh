#!/bin/bash
# Deployment retrain: best sweep config (c4) on train+val for a fixed 100 epochs (= mean best epoch of the
# early-stopped runs), 3 seeds. Test split stays untouched; compared against the train-only ensemble at the end.
cd /home/asl/Downloads/ASL/ASL_Translator
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=data/aslcitizen/trainval.log; PY="nice -n 19 .venv/bin/python"
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> $L; }
wait_resources() {
  while true; do
    set -- $($PY -c 'from src.sign.resources import free_ram_gb,free_vram_gb;print(int(free_ram_gb()), int(free_vram_gb()))')
    [ "$1" -ge 8 ] && [ "$2" -ge 2 ] && return; sleep 120
  done
}
CFG="--sources wlasl,citizen --min-train 10 --top-k 100000 --feat hands --derived --mirror 0.5 --bs 128 --d-model 192 --layers 2 --dropout 0.3 --wd 0.1 --trainval --epochs 100"
for s in 0 1 2; do
  out=checkpoints/sign/cit_tv/m$s
  for t in 1 2 3 4 5 6 7 8; do
    [ -f $out/results.json ] && break
    wait_resources; log "seed $s attempt $t"
    $PY -m src.sign.train $CFG --seed $s --out $out > data/aslcitizen/tv_$s.log 2>&1
    [ -f $out/results.json ] || { log "  seed $s failed: $(grep -a -m1 -o 'OutOfMemoryError' data/aslcitizen/tv_$s.log)"; sleep 300; }
  done
done
mkdir -p checkpoints/sign/cit_tv_ens
for s in 0 1 2; do cp checkpoints/sign/cit_tv/m$s/best.pt checkpoints/sign/cit_tv_ens/m$s.pt; done
LISTS=data/aslcitizen/vocab_demo.txt,data/aslcitizen/vocab_everyday.txt,data/wlasl/vocab_convo.txt
log "== train-only ensemble (deployed) =="; $PY scripts/eval_vocab.py checkpoints/sign/final --lists $LISTS >> $L 2>&1
log "== train+val ensemble =="; $PY scripts/eval_vocab.py checkpoints/sign/cit_tv_ens --lists $LISTS >> $L 2>&1
log "TRAINVAL DONE"
