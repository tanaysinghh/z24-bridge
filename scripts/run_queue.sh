#!/usr/bin/env bash
cd "$(dirname "$0")/.."
run_one() {
  model=$1; name=$2; shift 2
  log="logs/${name//\//_}.log"
  if [ -f "docs/results/${name}.json" ]; then echo "skip $name"; return; fi
  .venv/Scripts/python.exe -m src.train.train_${model} --run-name "$name" "$@" > "$log" 2>&1
  echo "done $name $(tail -4 "$log" | head -1)"
}
export -f run_one
grep -v '^\s*$' "$1" | xargs -P "${2:-3}" -L 1 bash -c 'run_one "$@"' _
