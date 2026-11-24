#!/bin/bash
cd /home/asl/Downloads/ASL/ASL_Translator
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
K=$1; PY="nice -n 19 .venv/bin/python"
run() { n=$1; shift; $PY -m src.sign.train --top-k $K --patience 40 --feat hands --out checkpoints/sign/sweep/k${K}_$n "$@" > data/wlasl/sweep_k${K}_$n.log 2>&1; }
run s96 --d-model 96 --layers 2 --dropout 0.3
run s64 --d-model 64 --layers 2 --dropout 0.3
run s128_l1 --d-model 128 --layers 1 --dropout 0.3
run s128_l3 --d-model 128 --layers 3 --dropout 0.3
run s128_d5 --d-model 128 --layers 2 --dropout 0.5
run s128_wd2 --d-model 128 --layers 2 --dropout 0.3 --wd 0.2
run s128_lr1 --d-model 128 --layers 2 --dropout 0.3 --lr 0.001
run s128_sm2 --d-model 128 --layers 2 --dropout 0.3 --smooth 0.2
run s128_seed1 --d-model 128 --layers 2 --dropout 0.3 --seed 1
run s128_seed2 --d-model 128 --layers 2 --dropout 0.3 --seed 2
echo SWEEP2 DONE > data/wlasl/sweep2_k$K.done
