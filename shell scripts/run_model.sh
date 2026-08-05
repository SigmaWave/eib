#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "Usage: ./run.sh <model-name>"
  exit 1
fi

MODEL="$1"

tmux new-session -d -s eib_run_${MODEL//:/_} "
{
python3 components/triplet_generator.py --model-name $MODEL --data-type semi_cleaned_data --text-column summary --start-date 2020-01-03 --end-date 2020-01-18 && \
python3 components/metrics_computator.py --triplets-path output/triplets_${MODEL}_semi_cleaned_data_summary.csv && \
python3 components/analyse_metrics.py output/triplets_${MODEL}_semi_cleaned_data_summary_JudgeLLM_metrics_computation.csv
} 2>&1 | tee tmux_logs.log
"
echo "Started tmux session for model: $MODEL"