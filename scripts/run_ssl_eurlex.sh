#!/usr/bin/env bash
gpuid=$1
seed=$2
data=EUR-Lex
head_label_num=800
echo '**************************** Self-Supervised Learning ******************************'
python ssl_main.py \
--gpuid $gpuid \
--data_dir ./data/$data \
--seed $seed \
--lr 5e-5 \
--encoder_lr 5e-5 \
--batch_size 16 \
--num_workers 4 \
--epochs 15 \
--patience 3 \
--head_label_num $head_label_num \
--num_workers 4 \
--bert_path ./bert/bert-base-uncased \
--save_path ./model_saved/$data

