#!/bin/bash
cd /home/asl/Downloads/ASL/ASL_Translator
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="nice -n 19 .venv/bin/python"; C="--top-k 9999 --vocab data/wlasl/vocab_convo.txt --feat hands --d-model 128 --layers 1 --dropout 0.3 --wd 0.2 --patience 40"
for s in 0 1; do
  $PY -m src.sign.train $C --seed $s --out checkpoints/sign/abl/base_$s > /dev/null 2>&1
  $PY -m src.sign.train $C --seed $s --mirror 0.5 --out checkpoints/sign/abl/mirror_$s > /dev/null 2>&1
  $PY -m src.sign.train $C --seed $s --derived --out checkpoints/sign/abl/derived_$s > /dev/null 2>&1
  $PY -m src.sign.train $C --seed $s --mirror 0.5 --derived --out checkpoints/sign/abl/both_$s > /dev/null 2>&1
done
echo done > checkpoints/sign/abl/DONE
