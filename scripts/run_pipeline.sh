#!/bin/bash
cd /home/asl/Downloads/ASL/ASL_Translator
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="nice -n 19 .venv/bin/python"
L=data/wlasl/pipeline.log
: > $L
echo "[$(date +%T)] extracting" >> $L

$PY scripts/wlasl_extract.py 4 > data/wlasl/extract.log 2>&1
echo "[$(date +%T)] extraction done: $(ls data/wlasl/landmarks | wc -l) files" >> $L
for K in 100 300; do
  echo "[$(date +%T)] training top-$K" >> $L
  $PY -m src.sign.train --top-k $K > data/wlasl/train_$K.log 2>&1
  tail -1 data/wlasl/train_$K.log >> $L
done
echo "[$(date +%T)] ALL DONE" >> $L
