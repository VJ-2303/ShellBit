#!/usr/bin/env bash
set -euo pipefail
config=${1:-config.yaml}
python3 prepare_data.py --config "$config"
python3 train_sft.py --config "$config"
data_dir=$(python3 -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["project"]["data_dir"])' "$config")
python3 eval_benchmark.py --config "$config" --test-file "$data_dir/challenge.jsonl"
python3 eval_benchmark.py --config "$config" --test-file "$data_dir/val.jsonl"
