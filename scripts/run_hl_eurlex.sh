#!/usr/bin/env bash
gpuid=$1
seed=$2
data=EUR-Lex
head_label_num=800
lr=0.001
swa=3
echo '**************************** Head Label Classifier Training ******************************'
python -m torch.distributed.launch --nproc_per_node=1 \
--master_port='1296' --master_addr='127.0.0.1' hl_main.py \
--gpuid $gpuid \
--data_dir ./data/$data \
--head_label_num $head_label_num \
--seed $seed \
--lr $lr \
--encoder_lr 5e-5 \
--batch_size 16 \
--valid_size 200 \
--swa_warmup $swa \
--num_workers 4 \
--epochs 100 \
--patience 3 \
--encoder_weights ./model_saved/$data/mlm_headlabel_${head_label_num}_encoder_lr_5e-05_lr_5e-05_swa_-1_fp16_False/BEST_encoder_checkpoint.pt \
--bert_path ./bert/bert-base-uncased \
--save_path ./model_saved/${data}