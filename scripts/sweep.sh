#!/bin/bash
# usage: sweep.sh TOPK  -> runs variants sequentially, each into checkpoints/sign/sweep/k<K>_<name>
cd /home/asl/Downloads/ASL/ASL_Translator
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
K=$1; PY="nice -n 19 .venv/bin/python"
run() { n=$1; shift; $PY -m src.sign.train --top-k $K --out checkpoints/sign/sweep/k${K}_$n "$@" > data/wlasl/sweep_k${K}_$n.log 2>&1; echo "k$K $n $(grep -a FINAL data/wlasl/sweep_k${K}_$n.log | cut -c7-)" >> data/wlasl/sweep_results.txt; }
run base
run noface --feat noface
run hands --feat hands
run small128 --d-model 128 --layers 2 --dropout 0.3
run small128_nf --d-model 128 --layers 2 --dropout 0.3 --feat noface
run small128_hands --d-model 128 --layers 2 --dropout 0.3 --feat hands
run drop4 --d-model 256 --layers 4 --dropout 0.4 --feat noface --wd 0.1
run mid192 --d-model 192 --layers 3 --dropout 0.35 --feat hands --wd 0.1
echo "k$K SWEEP DONE" >> data/wlasl/sweep_results.txt
