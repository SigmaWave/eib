#!/bin/bash
set -e

tmux new-session -d -s eib_run "
{
python3 components/triplet_generator.py --model-name llama3.1:8b --data-type semi_cleaned_data  --text-column summary --start-date 2020-01-03  --end-date 2020-01-18 && \
python3 components/triplet_generator.py --model-name mistral:7b --data-type semi_cleaned_data  --text-column summary --start-date 2020-01-03  --end-date 2020-01-18 && \
python components/metrics_computator.py  --triplets-path output/triplets_llama3.1:8b_semi_cleaned_data_summary.csv  && \
python components/metrics_computator.py  --triplets-path output/triplets_mistral:7b_semi_cleaned_data_summary.csv
} 2>&1 | tee tmux_logs.log
"
echo "Started tmux session"