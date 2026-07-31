#!/bin/bash
set -e

gcloud compute scp /Users/pierre/Desktop/EIB_local/database/nasdaq_semi_data_cleaned.csv \
  eib-central1a:~/eib/database/ \
  --zone=us-central1-a \
  --tunnel-through-iap