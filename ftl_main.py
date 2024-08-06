# -*- coding: utf-8 -*-
import os
# from apex import amp
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from deepxml.dataset import FTLDataset
from deepxml.models import XMLModel
from deepxml.modules import convert_weights
import torch.distributed as dist
import numpy as np

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--is_baseline', action='store_true')
	parser.add_argument('--ftl_learning', type=int, default=1)
	parser.add_argument('--local_rank', type=int, default=-1)
	parser.add_argument('--data_dir', help='path for the data folders')
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
	parser.add_argument('--swa_warmup', type=int, default=-1)
	parser.add_argument('--threshold', type=float, default=0.5,
						help='threshold for the evaluation (default: 0.5)')
	parser.add_argument('--test_model', action='store_true')
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
	parser.add_argument('--encoder_weights', default=None,
						help='weights from the encoder training')
	parser.add_argument('--encoder_fix',action='store_true')
	parser.add_argument('--toy', action='store_true')
	args = parser.parse_args()
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
	if not args.test_model:
		dist.init_process_group(backend='nccl',rank=args.local_rank)
		torch.cuda.set_device(args.local_rank)

	from transformers import BertTokenizer, BertModel, set_seed, AdamW
	from deepxml.networks import LWANetwork
	set_seed(args.seed)
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model_type = 'ftl_headlabel_{}_encoder_lr_{}_lr_{}_encoder_fix_{}_swa_{}_fp16_{}_seed_{}_bsl_{}'.\
		format(args.head_label_num,args.encoder_lr,args.lr,args.encoder_fix,args.swa_warmup,args.fp16,args.seed,args.is_baseline)
	args.model_type = model_type
	train_data_path = os.path.join(args.data_dir, 'xml_train_data_whead.json')
	test_data_path = os.path.join(args.data_dir, 'xml_test_data_whead.json')
	train_data = json.load(open(train_data_path, 'r'))
	test_data = json.load(open(test_data_path, 'r'))
	headlabel2id_path = os.path.join(args.data_dir, 'lm_label2id_head_label_{}.json'.format(args.head_label_num))
	headlabel2id = json.load(open(headlabel2id_path))
	label2id_path = os.path.join(args.data_dir, 'label2id.json')
	label2id = json.load(open(label2id_path, 'r'))
	if args.toy:
		train_data = train_data[:512]
		test_data = test_data[:512]
		args.patience = 1
		args.epochs = 5
		args.swa_warmup = 2
	print('train size', len(train_data))
	print('val size', len(test_data))
	print('head label num', args.head_label_num)
	print('label num', len(label2id))

	tokenizer = BertTokenizer.from_pretrained(args.bert_path)
	encoder = BertModel.from_pretrained(args.bert_path).to(device)
	args.tokenizer = tokenizer
	label_id_pattern = '[unused{}]'
	unused_size = 994
	if args.head_label_num > unused_size and not args.is_baseline:
		special_tokens = [label_id_pattern.format(i) for i in range(unused_size, args.head_label_num)]
		# special_tokens = [label_id_pattern.format(i) for i in range(unused_size, 13330)]
		special_tokens_dict = {'additional_special_tokens': special_tokens}
		args.tokenizer.add_special_tokens(special_tokens_dict)
		encoder.resize_token_embeddings(len(args.tokenizer))
	network = LWANetwork(bert_config=encoder.config, num4label=len(label2id)).to(device)

	train_dataset = FTLDataset(data=train_data,tokenzier=args.tokenizer,label2id=label2id,headlabel2id=headlabel2id,
							   maxlength=args.maxlength,train=True,
							  is_baseline=args.is_baseline)
	test_dataset = FTLDataset(data=test_data,tokenzier=args.tokenizer,label2id=label2id,headlabel2id=headlabel2id,
							  maxlength=args.maxlength,train=True,
							is_baseline=args.is_baseline)

	if not args.test_model:
		train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
		train_loader = DataLoader(train_dataset,
								  batch_size=args.batch_size,
								  num_workers=args.num_workers,
								  pin_memory=True,
								  shuffle=False,
								  drop_last=False,
								  sampler=train_sampler)
	
	test_loader = DataLoader(test_dataset,
							batch_size=args.batch_size,
							num_workers=args.num_workers,
							pin_memory=True,
							shuffle=False,
							drop_last=False)

	if not args.test_model:
		optimizer_grouped_parameters = [
			{'params': encoder.parameters(), 'lr': args.encoder_lr},
			{'params': network.parameters(), 'lr': args.lr}
		]
		optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)

		if not args.is_baseline:	
			if args.encoder_weights is not None:
				print('load hl encoder weights')
				# encoder.load_state_dict(convert_weights(torch.load(args.encoder_weights)))
				encoder.load_state_dict(convert_weights(torch.load(args.encoder_weights, map_location="cpu")))
				encoder = encoder.to(device)
		else:
			print('Running Baseline. Without loading encoder_weights ! ')

		# print('--------------------- embeddings are fixed in step2 !! ----------------\n\n')
		# for name, param in encoder.named_parameters():
		# 	if 'embeddings' in name:
		# 		param.requires_grad = False

		if args.encoder_fix:
			for name, param in encoder.named_parameters():
				param.requires_grad = False
				if 'pooler' in name:
					param.requires_grad = True
					break

		if args.fp16:
			print('Using 16-Bit (mixed) Precision')
			[encoder, network], optimizer = amp.initialize([encoder, network], optimizer, opt_level=args.fp16_opt_level)
		# encoder = nn.DataParallel(encoder)
		# network = nn.DataParallel(network)
		encoder=nn.parallel.DistributedDataParallel(encoder,
												device_ids=[args.local_rank],
												output_device=args.local_rank,
												find_unused_parameters=True).to(device)
		network=nn.parallel.DistributedDataParallel(network,
												device_ids=[args.local_rank],
												output_device=args.local_rank,
												find_unused_parameters=True).to(device)
		model = XMLModel(args)
		model.train(encoder=encoder,network=network,optimizer=optimizer,
					train_loader=train_loader,valid_loader=test_loader,device=device)
	else:
		args.local_rank = 0
		args.log_path = None
		train_labels = [[label2id[x] 
			for x in  item['labels']] 
				for item in train_data]
		

		save_path = args.save_path
		encoder_weights = os.path.join(save_path, 'BEST_encoder_checkpoint.pt')
		network_weights = os.path.join(save_path, 'BEST_xml_checkpoint.pt')
		model = XMLModel(args)
		encoder = model.load_model(encoder, encoder_weights)
		network = model.load_model(network, network_weights)
		outputs = model.predict(encoder,network,test_loader,device,step2eval=False)
		# model.eval(outputs[0], outputs[1])
		print('\nBelow is PSP scores')
		# train_label_path = os.path.join(args.data_dir,'train_labels.npy')
		# with open(train_label_path) as fp:
		# 	train_labels = np.load(train_label_path, allow_pickle=True)
		
		model.eval_psp(outputs[0], outputs[1], train_labels)

if __name__ == '__main__':
	main()



