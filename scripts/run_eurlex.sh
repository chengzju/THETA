#!/usr/bin/env bash
gpuid=$1
seed=$2

sh ./scripts/run_preprocess_eurlex.sh

sh ./scripts/run_ssl_eurlex.sh $gpuid $seed

sh ./scripts/run_hl_eurlex.sh $gpuid $seed

sh ./scripts/run_ftl_eurlex.sh $gpuid $seed
