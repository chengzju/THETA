# -*- coding: utf-8 -*-
import os
import json
import argparse
from apex import amp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from deepxml.dataset import MLMDataset
from deepxml.models import MLMModel
import numpy as np

import torch.distributed as dist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',help='path for the data folders')
    parser.add_argument('--config_path', help='path of data configure yaml')
    parser.add_argument('--bert_path', help='path for the bert model')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', default=5e-5, type=float, help='classifier learning rate')
    parser.add_argument('--encoder_lr', default=5e-5, type=float, help='encoder learning_rate')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--gpuid', default='0', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--save_path', help='path for the model checkpoints folders')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--head_label_num', type=int, default=50)
    parser.add_argument('--maxlength', type=int, default=503)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--swa_warmup', type=int, default=-1)
    parser.add_argument('--encoder_fix',action='store_true')
    parser.add_argument(
        "--fp16", action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.",
    )
    parser.add_argument('--toy', action='store_true')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    dist.init_process_group(backend='nccl',rank=args.local_rank)
    torch.cuda.set_device(args.local_rank)

    from transformers import BertTokenizer, BertModel, set_seed, AdamW
    from deepxml.networks import MLMNetwork
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = 'mlm_headlabel_{}_encoder_lr_{}_lr_{}_swa_{}_fp16_{}'.format(args.head_label_num,
                                                                              args.encoder_lr,
                                                                              args.lr,
                                                                              args.swa_warmup,
                                                                              args.fp16)
    args.model_type = model_type
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    args.tokenizer = tokenizer
    encoder = BertModel.from_pretrained(args.bert_path).to(device)
    unused_size = 994
    label_id_pattern = '[unused{}]'
    if args.head_label_num > unused_size:
        special_tokens = [label_id_pattern.format(i) for i in range(unused_size, args.head_label_num)]
        special_tokens_dict = {'additional_special_tokens': special_tokens}
        args.tokenizer.add_special_tokens(special_tokens_dict)
        encoder.resize_token_embeddings(len(args.tokenizer))

    network= MLMNetwork(bert_config=encoder.config).to(device)
    optimizer_grouped_parameters = [{'params': encoder.parameters(), 'lr': args.encoder_lr},
                                    {'params': network.parameters(), 'lr': args.lr}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    if args.fp16:
        print('Using 16-Bit (mixed) Precision')
        [encoder, network], optimizer = amp.initialize([encoder, network], optimizer, opt_level=args.fp16_opt_level)
    # encoder = nn.DataParallel(encoder)
    # network = nn.DataParallel(network)
    model = MLMModel(args)

    train_data_path = os.path.join(args.data_dir,
                                       'mlm_train_data_head_label_{}.json'.format(
                                                 args.head_label_num))
    train_data = json.load(open(train_data_path,'r'))
    if args.toy:
        train_data = train_data[:512]
        args.patience = 1
        args.epochs = 5
        args.swa_warmup = 2
    print('train size', len(train_data))
    print('head label num', args.head_label_num)
    train_dataset = MLMDataset(data=train_data, tokenizer=tokenizer,maxlength=args.maxlength)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              shuffle=False,
                              drop_last=False,
                              sampler=train_sampler
                              )

    test_data = list(np.array(train_data)[np.random.permutation(len(train_data))[:50000]])
    test_dataset = MLMDataset(data=test_data, tokenizer=tokenizer,maxlength=args.maxlength)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             shuffle=False,
                             drop_last=False
                             )

    encoder=nn.parallel.DistributedDataParallel(encoder,
                                                device_ids=[args.local_rank],
                                                output_device=args.local_rank,
                                                find_unused_parameters=True).to(device)
    network=nn.parallel.DistributedDataParallel(network,
                                                device_ids=[args.local_rank],
                                                output_device=args.local_rank,
                                                find_unused_parameters=True).to(device)

    # if args.encoder_fix:
    #     print('Encoder is fixed, without fix embeddings !!')
    #     # fix bert encoder, only train embeddings
    #     for name, param in encoder.named_parameters():
    #         param.requires_grad = False

    #         if 'embeddings' in name:
    #             param.requires_grad = True
    #         if 'pooler' in name:
    #             param.requires_grad = True
    # else:
    #     print('Encoder is unfixed !!')


    model.train(encoder=encoder, network=network, optimizer=optimizer, train_loader=train_loader,
                valid_loader=test_loader, device=device)

if __name__ == '__main__':
    main()






