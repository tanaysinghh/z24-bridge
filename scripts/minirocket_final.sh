#!/usr/bin/env bash
cd "$(dirname "$0")/.."
D=z24_ch22-26_w2000_s1000
PY=.venv/Scripts/python.exe
for s in 42 1 2; do
  $PY -m src.train.train_minirocket --data $D --alpha-mode loso --seed $s --run-name final/minirocket_s$s --eval-test --n-jobs 8
done
$PY -m src.train.train_minirocket --data $D --alpha-mode loso --seed 42 --run-name oof/minirocket_f1 --train-setups 3 4 5 6 --n-jobs 8
$PY -m src.train.train_minirocket --data $D --alpha-mode loso --seed 42 --run-name oof/minirocket_f2 --train-setups 1 2 5 6 --n-jobs 8
$PY -m src.train.train_minirocket --data $D --alpha-mode loso --seed 42 --run-name oof/minirocket_f3 --train-setups 1 2 3 4 --n-jobs 8
