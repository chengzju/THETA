#!/usr/bin/env bash
gpuid=$1
seed=$2
data=EUR-Lex
head_label_num=800
lr=5e-4
swa=5
hl_path=hl_headlabel_${head_label_num}_encoder_lr_5e-05_lr_0.001_encoder_fix_False_swa_3_fp16_False_seed_${seed}
echo '**************************** Head Label Predicting ******************************'
python hl_main.py \
--test_model \
--gpuid $gpuid \
--data_dir ./data/$data \
--head_label_num $head_label_num \
--seed $seed \
--lr $lr \
--encoder_lr 5e-5 \
--batch_size 16 \
--swa_warmup $swa \
--num_workers 4 \
--epochs 100 \
--patience 3 \
--bert_path ./bert/bert-base-uncased \
--save_path ./model_saved/$data/$hl_path

echo '**************************** Fulfilling Tail Label Classifiers Training ******************************'
python ftl_main.py \
--gpuid $gpuid \
--data_dir ./data/$data \
--head_label_num $head_label_num \
--seed $seed \
--lr $lr \
--encoder_lr 5e-5 \
--batch_size 16 \
--swa_warmup $swa \
--num_workers 4 \
--epochs 100 \
--patience 3 \
--encoder_weights ./model_saved/$data/$hl_path/BEST_encoder_checkpoint.pt \
--bert_path ./bert/bert-base-uncased \
--save_path ./model_saved/$data