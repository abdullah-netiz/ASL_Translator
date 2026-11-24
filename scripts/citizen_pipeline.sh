#!/bin/bash
# Unattended: config sweep (select by VAL loss) -> 3-seed final ensemble -> eval.
# Shared-PC safe: before every run waits for free RAM and VRAM; retries a run that crashed (e.g. CUDA OOM
# caused by other users' jobs) instead of moving on with a missing result.
cd /home/asl/Downloads/ASL/ASL_Translator
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=data/aslcitizen/pipeline.log; PY="nice -n 19 .venv/bin/python"
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> $L; }
until grep -q "^done" data/aslcitizen/extract.log; do sleep 60; done

wait_resources() {   # need >=8 GB free RAM and >=3 GB free VRAM
  while true; do
    r=$($PY -c 'from src.sign.resources import free_ram_gb,free_vram_gb;print(int(free_ram_gb()), int(free_vram_gb()))')
    set -- $r; [ "$1" -ge 8 ] && [ "$2" -ge 3 ] && return; sleep 120
  done
}
run() {  # run <out_dir> <logfile> <args...>; retries until results.json exists (max 8 tries)
  out=$1; lf=$2; shift 2
  for t in 1 2 3 4 5 6 7 8; do
    [ -f $out/results.json ] && return 0
    wait_resources
    $PY -m src.sign.train "$@" --out $out > $lf 2>&1
    [ -f $out/results.json ] && return 0
    log "  $out attempt $t failed: $(grep -a -m1 -o 'OutOfMemoryError\|Error[^\n]*' $lf | head -1) - retrying in 5 min"
    sleep 300
  done
  return 1
}

BASE="--sources wlasl,citizen --min-train 10 --top-k 100000 --feat hands --derived --mirror 0.5 --epochs 150 --patience 12 --bs 128"
declare -A CFG=(
  [c1]="--d-model 256 --layers 2 --dropout 0.2 --wd 0.05"
  [c2]="--d-model 256 --layers 4 --dropout 0.2 --wd 0.05"
  [c3]="--d-model 384 --layers 4 --dropout 0.25 --wd 0.05"
  [c4]="--d-model 192 --layers 2 --dropout 0.3 --wd 0.1"
)
for n in c1 c2 c3 c4; do
  log "sweep $n: ${CFG[$n]}"
  run checkpoints/sign/cit/$n data/aslcitizen/train_$n.log $BASE ${CFG[$n]}
  log "$n: $(grep -a FINAL data/aslcitizen/train_$n.log | grep -o "'best_epoch.*val_top1_at_best': [0-9.]*")"
done
BEST=$($PY - <<'PY'
import json,glob,os
r=[(json.load(open(f))['best_val_loss'],os.path.basename(os.path.dirname(f))) for f in glob.glob('checkpoints/sign/cit/c*/results.json')]
print(min(r)[1])
PY
)
[ -z "$BEST" ] && { log "NO SWEEP RESULTS - abort"; exit 1; }
log "best config by val loss: $BEST (${CFG[$BEST]})"
for s in 1 2; do
  run checkpoints/sign/cit_ens/m$s data/aslcitizen/final_$s.log $BASE ${CFG[$BEST]} --seed $s
  log "seed $s done"
done
mkdir -p checkpoints/sign/cit_ens/m0 && cp checkpoints/sign/cit/$BEST/best.pt checkpoints/sign/cit_ens/m0/best.pt
$PY scripts/eval_vocab.py checkpoints/sign/cit_ens >> $L 2>&1
log "PIPELINE DONE"
