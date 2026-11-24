#!/bin/bash
cd /home/asl/Downloads/ASL/ASL_Translator
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

PY="nice -n 19 .venv/bin/python"; C="--top-k 9999 --vocab data/wlasl/vocab_convo.txt --feat hands --d-model 128 --layers 1 --dropout 0.3 --wd 0.2"
for s in 0 1 2 3 4; do
  $PY -m src.sign.train $C --seed $s --patience 40 --out checkpoints/sign/ens_val/m$s > data/wlasl/final_val_$s.log 2>&1
done
for s in 0 1 2 3 4; do
  $PY -m src.sign.train $C --seed $s --trainval --epochs 240 --out checkpoints/sign/ens_tv/m$s > data/wlasl/final_tv_$s.log 2>&1
done
echo done > data/wlasl/final.done
