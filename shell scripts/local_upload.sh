#!/bin/bash
set -e

# ./local_upload.sh filepath
gcloud compute scp "$1" \
  eib-central1a:~/ \
  --zone=us-central1-a \
  --tunnel-through-iap
