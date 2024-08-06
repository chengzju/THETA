# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   Author : theta
   date : 2021/2/2
-------------------------------------------------
"""
import os
import argparse
from transformers import BertTokenizer
from deepxml.data_utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='path for the data folders')
    parser.add_argument('--config_path', help='path of data configure yaml')
    parser.add_argument('--bert_path', help='path for the bert model')
    parser.add_argument('--head_label_num', type=int, default=50)
    parser.add_argument('--maxlength', type=int, default=500)
    parser.add_argument('--mask_rate', type=float, default=0.15)
    parser.add_argument('--num_workers', default=-1, type=int)
    parser.add_argument('--text_repeat', action='store_true')
    parser.add_argument('--justified', action='store_true')
    args = parser.parse_args()
    print('load data from ', args.data_dir)
    train_data_output_path = os.path.join(args.data_dir, 'train_data.json')
    label_output_path = os.path.join(args.data_dir, 'label_freq.json'.format(args.head_label_num))
    train_data_path = os.path.join(args.data_dir, 'train_texts.txt')
    train_label_path = os.path.join(args.data_dir, 'train_labels.txt')
    train_data, freq4labels = load_data(data_path=train_data_path, label_path=train_label_path,
                                        data_output=train_data_output_path, label_output=label_output_path)

    test_data_output_path = os.path.join(args.data_dir, 'test_data.json')
    test_data_path = os.path.join(args.data_dir, 'test_texts.txt')
    test_label_path = os.path.join(args.data_dir, 'test_labels.txt')
    test_data, _ = load_data(data_path=test_data_path, label_path=test_label_path,
                             data_output=test_data_output_path)

    labels = sorted(freq4labels.items(), key=lambda x: x[1], reverse=True)
    print('labels num', len(labels))
    labels = [x[0] for x in labels]
    label2id = {j: i for i, j in enumerate(labels)}
    label2id_path = os.path.join(args.data_dir, 'label2id.json')
    json.dump(label2id, open(label2id_path, 'w'), indent=4, ensure_ascii=False)

    label_id_pattern = '[unused{}]'
    unused_size = 994
    if args.head_label_num > unused_size:
        add_id_st = 30522
        add_tokens = [label_id_pattern.format(i) for i in range(unused_size, args.head_label_num)]
        add_tokens2id = {j: add_id_st + i for i, j in enumerate(add_tokens)}
        headlabels = labels[:unused_size]
        addlabels = labels[unused_size:args.head_label_num]
    else:
        add_tokens2id = None
        headlabels = labels[:args.head_label_num]
        addlabels = []

    lm_label2id = {j: label_id_pattern.format(i) for i, j in enumerate(headlabels+addlabels)}
    lm_label2id_path = os.path.join(args.data_dir, 'lm_label2id_head_label_{}.json'.format(args.head_label_num))
    json.dump(lm_label2id, open(lm_label2id_path, 'w'), indent=4, ensure_ascii=False)

    print('get mlm data')
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    headlabels_list = [headlabels, addlabels]
    mlmd = MLMDataset(tokenizer, lm_label2id, headlabels_list, mask_rate=args.mask_rate,
                    maxlength=args.maxlength, text_repeat=args.text_repeat)
    train_data4mlm_output_path = os.path.join(args.data_dir,
                                             'mlm_train_data_head_label_{}.json'.format(
                                                 args.head_label_num))
    if args.num_workers <= 1:
        train_data4mlm = mlmd.process(train_data, output=train_data4mlm_output_path, add_label2id=add_tokens2id)
    else:
        train_data4mlm = mlmd.multiprocess(train_data, output=train_data4mlm_output_path, add_label2id=add_tokens2id,
                                         process_num=args.num_workers)
    print('mlm train size', len(train_data4mlm))

    print('get xml data')
    train_data4xml_output_path = os.path.join(args.data_dir,
                                              'xml_train_data.json')
    test_data4xml_output_path = os.path.join(args.data_dir,
                                             'xml_test_data.json')
    if args.num_workers <= 1:
        train_data4xml = xml_sentence_segment(train_data, tokenizer, maxlength=args.maxlength, output=train_data4xml_output_path)
        test_data4xml = xml_sentence_segment(test_data, tokenizer, maxlength=args.maxlength, output=test_data4xml_output_path)
    else:
        train_data4xml = multi_xml_sentence_segment(train_data, tokenizer, maxlength=args.maxlength,
                                                    output=train_data4xml_output_path, process_num=args.num_workers)
        test_data4xml = multi_xml_sentence_segment(test_data, tokenizer, maxlength=args.maxlength,
                                                   output=test_data4xml_output_path, process_num=args.num_workers)
    print('xml train size', len(train_data4xml))
    print('xml test size', len(test_data4xml))

if __name__ == '__main__':
    main()




