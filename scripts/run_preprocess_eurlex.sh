#!/usr/bin/env bash
data=EUR-Lex
head_label_num=800
echo '**************************** Data Preprocessing ******************************'
python preprocess.py \
--data_dir ./data/$data \
--bert_path ./bert/bert-base-uncased \
--head_label_num $head_label_num \
--maxlength 500 \
--num_workers 4 \
--text_repeat


