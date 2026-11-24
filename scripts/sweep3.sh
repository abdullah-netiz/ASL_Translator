#!/bin/bash
# conversational vocab runs; waits for sweep2
cd /home/asl/Downloads/ASL/ASL_Translator
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
until [ -f data/wlasl/sweep2_k100.done ]; do sleep 10; done
PY="nice -n 19 .venv/bin/python"
for V in convo convo_plus; do
  F=data/wlasl/vocab_$V.txt
  for cfg in "l1 --d-model 128 --layers 1 --dropout 0.3" "l1wd2 --d-model 128 --layers 1 --dropout 0.3 --wd 0.2" "l2wd2 --d-model 128 --layers 2 --dropout 0.3 --wd 0.2" "l1_192 --d-model 192 --layers 1 --dropout 0.4 --wd 0.1"; do
    set -- $cfg; n=$1; shift
    $PY -m src.sign.train --top-k 9999 --vocab $F --patience 40 --feat hands --out checkpoints/sign/sweep/k${V}_$n "$@" > data/wlasl/sweep_k${V}_$n.log 2>&1
  done
done
echo done > data/wlasl/sweep3.done
