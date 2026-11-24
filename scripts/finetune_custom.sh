#!/bin/bash
# After recording your own clips with scripts/record_signs.py, run:  bash scripts/finetune_custom.sh
# Retrains the deployed configuration (WLASL + ASL Citizen, 2,303 words) on train+val PLUS your clips
# (oversampled x4) -> checkpoints/sign/final_custom. The server prefers final_custom when it exists;
# delete that folder to go back to checkpoints/sign/final. Takes ~3 h (3 models x 100 epochs) on the shared GPU.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="nice -n 19 .venv/bin/python"
n=$(ls data/wlasl/custom/*__*.npy 2>/dev/null | wc -l); [ "$n" -eq 0 ] && { echo "no clips in data/wlasl/custom - record some first"; exit 1; }
echo "$n custom clips found"
wait_resources() {   # shared PC: need >= 8 GB free RAM and >= 2 GB free VRAM
  while true; do
    set -- $($PY -c 'from src.sign.resources import free_ram_gb,free_vram_gb;print(int(free_ram_gb()), int(free_vram_gb()))')
    [ "$1" -ge 8 ] && [ "$2" -ge 2 ] && return; echo "waiting for free RAM/VRAM ($1 GB / $2 GB)"; sleep 120
  done
}
C="--sources wlasl,citizen --min-train 10 --top-k 100000 --feat hands --derived --mirror 0.5 --bs 128 --d-model 192 --layers 2 --dropout 0.3 --wd 0.1 --trainval --epochs 100 --custom"
rm -rf checkpoints/sign/_custom_tmp; mkdir -p checkpoints/sign/_custom_tmp
for s in 0 1 2; do
  for t in 1 2 3 4 5; do                      # retry if another user's job causes CUDA OOM
    [ -f checkpoints/sign/_custom_tmp/m$s/results.json ] && break
    wait_resources
    $PY -m src.sign.train $C --seed $s --out checkpoints/sign/_custom_tmp/m$s > data/wlasl/custom_$s.log 2>&1
  done
  [ -f checkpoints/sign/_custom_tmp/m$s/results.json ] || { echo "seed $s failed - see data/wlasl/custom_$s.log"; exit 1; }
done
rm -rf checkpoints/sign/final_custom; mkdir -p checkpoints/sign/final_custom
for s in 0 1 2; do cp checkpoints/sign/_custom_tmp/m$s/best.pt checkpoints/sign/final_custom/m$s.pt; done
echo "done -> checkpoints/sign/final_custom (restart the server to use it)"
