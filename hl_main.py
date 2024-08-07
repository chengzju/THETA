# -*- coding: utf-8 -*-
import os
import argparse
import json
import numpy as np
# from apex import amp
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from deepxml.dataset import HLDataset
from deepxml.models import XMLModel, DataStatistics
from deepxml.modules import convert_weights
import torch.distributed as dist

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--local_rank', type=int, default=0)
	parser.add_argument('--data_dir', help='path for the data folders')
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
	parser.add_argument('--maxlength', type=int, default=502)
	parser.add_argument('--swa_warmup', type=int, default=-1)
	parser.add_argument('--threshold', type=float, default=0.5,
						help='threshold for the evaluation (default: 0.5)')
	parser.add_argument('--test_model', action='store_true')
	parser.add_argument('--use_ddp', action='store_true')
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
	parser.add_argument('--valid_size', type=int, default=200)
	parser.add_argument('--toy', action='store_true')
	parser.add_argument('--encoder_fix',action='store_true')
	args = parser.parse_args()
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
	if not args.test_model and args.use_ddp:
		dist.init_process_group(backend='nccl',rank=args.local_rank)
		torch.cuda.set_device(args.local_rank)

	from transformers import BertTokenizer, BertModel,set_seed, AdamW
	from deepxml.networks import LWANetwork
	set_seed(args.seed)
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model_type = 'hl_headlabel_{}_encoder_lr_{}_lr_{}_encoder_fix_{}_swa_{}_fp16_{}_seed_{}'.format(args.head_label_num,
																									args.encoder_lr,
																									args.lr,
																									args.encoder_fix,
																									args.swa_warmup,
																									args.fp16,
																									args.seed)
	args.model_type = model_type
	tokenizer = BertTokenizer.from_pretrained(args.bert_path)
	encoder = BertModel.from_pretrained(args.bert_path).to(device)
	args.tokenizer = tokenizer
	label_id_pattern = '[unused{}]'
	unused_size = 994
	if args.head_label_num > unused_size:
		special_tokens = [label_id_pattern.format(i) for i in range(unused_size, args.head_label_num)]
		# special_tokens = [label_id_pattern.format(i) for i in range(unused_size, 13330)]
		special_tokens_dict = {'additional_special_tokens': special_tokens}
		args.tokenizer.add_special_tokens(special_tokens_dict)
		encoder.resize_token_embeddings(len(args.tokenizer))
	network = LWANetwork(bert_config=encoder.config, num4label=args.head_label_num).to(device)

	train_data_path = os.path.join(args.data_dir,'xml_train_data.json')
	test_data_path = os.path.join(args.data_dir, 'xml_test_data.json')
	train_data = json.load(open(train_data_path, 'r'))
	test_data = json.load(open(test_data_path, 'r'))
	label2id_path = os.path.join(args.data_dir, 'label2id.json')
	label2id = json.load(open(label2id_path, 'r'))
	label2id = {k: v for k, v in label2id.items() if v < args.head_label_num}
	if args.toy:
		# train_data = train_data[:512]
		# test_data = test_data[:512]
		toy_size = 50000
		train_idx = np.random.permutation(len(train_data))[:toy_size]
		train_data = np.array(train_data)[train_idx]
		test_idx = np.random.permutation(len(test_data))[:toy_size]
		test_data = np.array(test_data)[test_idx]
		args.patience = 1
		args.epochs = 5
		args.swa_warmup = 2
	if not args.test_model:

		# test_size = len(test_data)
		# rand_indices = np.random.permutation(test_size)
		# test_data = np.array(test_data)
		# test_data = test_data[rand_indices[:50000]]

		if args.valid_size > 0:
			data_num = len(train_data)
			np.random.seed(args.seed)
			rand_indices = np.random.permutation(data_num)
			train_data = np.array(train_data)
			test_data = train_data[rand_indices[:args.valid_size]]
			train_data = train_data[rand_indices[args.valid_size:]]
		print('train size', len(train_data))
		print('test size', len(test_data))
		print('head label num', args.head_label_num)
		train_dataset = HLDataset(data=train_data, tokenizer=tokenizer, label2id=label2id,
										 maxlength=args.maxlength, train=True)
		test_dataset = HLDataset(data=test_data, tokenizer=tokenizer,label2id=label2id,
								 maxlength=args.maxlength,train=True)
		if args.use_ddp:
			train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
			train_loader = DataLoader(train_dataset,
									  batch_size=args.batch_size,
									  num_workers=args.num_workers,
									  pin_memory=True,
									  shuffle=False,
									  sampler=train_sampler,
									  drop_last=False)
		else:
			train_loader = DataLoader(train_dataset,
									  batch_size=args.batch_size,
									  num_workers=args.num_workers,
									  pin_memory=True,
									  shuffle=True,
									  drop_last=False)
		test_loader = DataLoader(test_dataset,
								 batch_size=args.batch_size,
								 num_workers=args.num_workers,
								 pin_memory=True,
								 shuffle=False,
								 drop_last=False)
		if args.encoder_weights is not None:
			print('load mlm encoder weights')
			# encoder.load_state_dict(convert_weights(torch.load(args.encoder_weights)))
			encoder.load_state_dict(convert_weights(torch.load(args.encoder_weights, map_location="cpu")))
			encoder = encoder.to(device)

		# print('All layers freezed -----------------****************************')
		# all_layers=['layer.0', 'layer.1', 'layer.2', 'layer.3','layer.4', 'layer.5', 'layer.6', 'layer.7','layer.8', 'layer.9', 'layer.10', 'layer.11','pooler']
		# unfreeze_layers=all_layers[11+1: ]
		# for name, param in encoder.named_parameters():
		# 	param.requires_grad = False
		# 	for ele in unfreeze_layers:
		# 		if ele in name:
		# 			param.requires_grad = True
		# 			break

		if args.encoder_fix:
			for name, param in encoder.named_parameters():
				if 'embeddings' in name:
					param.requires_grad = False

		optimizer_grouped_parameters = [{'params': encoder.parameters(), 'lr': args.encoder_lr},
										{'params': network.parameters(), 'lr': args.lr}]
		optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
		if args.fp16:
			print('Using 16-Bit (mixed) Precision')
			[encoder, network], optimizer = amp.initialize([encoder, network], optimizer, opt_level=args.fp16_opt_level)
		if args.use_ddp:
			encoder=nn.parallel.DistributedDataParallel(encoder,
													device_ids=[args.local_rank],
													output_device=args.local_rank,
													find_unused_parameters=True).to(device)
			network=nn.parallel.DistributedDataParallel(network,
													device_ids=[args.local_rank],
													output_device=args.local_rank,
													find_unused_parameters=True).to(device)
		else:
			encoder = nn.DataParallel(encoder)
			network = nn.DataParallel(network)
		model = XMLModel(args)
		# model = DataStatistics(args)
		model.train(encoder=encoder, network=network, optimizer=optimizer,train_loader=train_loader,
					valid_loader=test_loader, device=device)
	else:
		print('Generating data')
		args.log_path = None
		print('train size', len(train_data))
		print('test size', len(test_data))
		print('head label num', args.head_label_num)
		train_dataset = HLDataset(data=train_data, tokenizer=tokenizer, label2id=label2id,
								  maxlength=args.maxlength, train=True)
		test_dataset = HLDataset(data=test_data, tokenizer=tokenizer, label2id=label2id,
								 maxlength=args.maxlength, train=True)
		train_loader = DataLoader(train_dataset,
								  batch_size=args.batch_size,
								  num_workers=args.num_workers,
								  pin_memory=True,
								  shuffle=False,
								  drop_last=False)
		test_loader = DataLoader(test_dataset,
								 batch_size=args.batch_size,
								 num_workers=args.num_workers,
								 pin_memory=True,
								 shuffle=False,
								 drop_last=False)

		savepath = args.save_path
		print('hl path', savepath)
		model = XMLModel(args)
		encoder_weights = os.path.join(savepath, 'BEST_encoder_checkpoint.pt')
		network_weights = os.path.join(savepath, 'BEST_xml_checkpoint.pt')
		# encoder = model.load_model(encoder, encoder_weights)
		# network = model.load_model(network, network_weights)
		encoder.load_state_dict(convert_weights(torch.load(encoder_weights, map_location="cpu")))
		encoder = encoder.to(device)  
		network.load_state_dict(convert_weights(torch.load(network_weights, map_location="cpu")))
		network = network.to(device)    
		if args.fp16:
			print('Using 16-Bit (mixed) Precision')
			[encoder, network] = amp.initialize([encoder, network],
												opt_level=args.fp16_opt_level)
		encoder = nn.DataParallel(encoder)
		network = nn.DataParallel(network)

		k = 200 # top 50 pred labels

		train_data = list(train_data)
		test_data = list(test_data)
		train_data_whead_path = os.path.join(args.data_dir,'xml_train_data_whead.json')
		test_data_whead_path = os.path.join(args.data_dir, 'xml_test_data_whead.json')


		print('train data')
		outputs = model.predict(encoder, network, train_loader, device)
		# model.eval(outputs[0], outputs[1])
		for i,item in enumerate(tqdm(train_data,desc='Train Data')):
			logits = outputs[1][i]
			pred_labels = []
			topk = np.argpartition(-logits, k)[:k]
			resort_k = np.argsort(-logits[topk])

			for j in range(k):
				if logits[topk[resort_k[j]]] > 0:
					pred_labels.append(topk[resort_k[j]])
				else:
					break
			item['pred_labels'] = [int(x) for x in pred_labels]
		json.dump(train_data, open(train_data_whead_path, 'w'), indent=4, ensure_ascii=False)

		print('test data')
		outputs = model.predict(encoder, network, test_loader, device)
		# model.eval(outputs[0], outputs[1])
		for i, item in enumerate(tqdm(test_data, desc='Test Data')):
			logits = outputs[1][i]
			pred_labels = []
			topk = np.argpartition(-logits, k)[:k]
			resort_k = np.argsort(-logits[topk])

			for j in range(k):
				if logits[topk[resort_k[j]]] > 0:
					pred_labels.append(topk[resort_k[j]])
				else:
					break
			item['pred_labels'] = [int(x) for x in pred_labels]
		json.dump(test_data, open(test_data_whead_path, 'w'), indent=4, ensure_ascii=False)

		# train_pred_path = os.path.join(args.data_dir,'train_pred_labels.json')
		# train_pred_labels = []
		# for i,item in enumerate(tqdm(train_data,desc='Train Data')):
		# 	train_pred_labels.append(item['pred_labels'])
		# json.dump(train_pred_labels, open(train_pred_path, 'w'), indent=4, ensure_ascii=False)

		# test_pred_path = os.path.join(args.data_dir, 'test_pred_labels.json')
		# test_pred_labels = []
		# for i, item in enumerate(tqdm(test_data, desc='Test Data')):
		# 	test_pred_labels.append(item['pred_labels'])
		# json.dump(test_pred_labels, open(test_pred_path, 'w'), indent=4, ensure_ascii=False)


if __name__ == '__main__':
	main()



