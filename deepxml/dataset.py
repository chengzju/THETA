# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.utils.data import Dataset

class MLMDataset(Dataset):
    def __init__(self,data,tokenizer, maxlength=503):
        self.data = data
        self.tokenizer = tokenizer
        self.maxlength = maxlength

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = item['input_ids']
        output_mask_idx = item['output_mask_idx']
        _output_ids = item['output_ids']
        output_mask = []
        output_ids = []
        if len(output_mask_idx) > 0:
            for i in range(len(input_ids)):
                if i == output_mask_idx[0]:
                    output_mask.append(1)
                    output_ids.append(_output_ids.pop(0))
                    output_mask_idx.pop(0)
                else:
                    output_mask.append(0)
                    output_ids.append(self.tokenizer.pad_token_id)
                if len(output_mask_idx) == 0:
                    break
        output_mask += [0] * (len(input_ids) - len(output_mask))
        output_ids += [self.tokenizer.pad_token_id] * (len(input_ids) - len(output_mask))
        input_mask = [1] * len(input_ids)
        input_type_ids = [0] * len(input_ids)
        for i in range(input_ids.index(self.tokenizer.sep_token_id) + 1, len(input_ids)):
            input_type_ids[i] = 1

        input_ids += [self.tokenizer.pad_token_id] * (self.maxlength - len(input_ids))
        input_type_ids += [self.tokenizer.pad_token_type_id] * (self.maxlength - len(input_type_ids))
        input_mask += [0] * (self.maxlength - len(input_mask))
        output_mask += [0] * (self.maxlength - len(output_mask))
        output_ids += [self.tokenizer.pad_token_id] * (self.maxlength - len(output_ids))

        input_ids = torch.tensor(input_ids,dtype=torch.long)
        input_type_ids = torch.tensor(input_type_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        output_mask = torch.tensor(output_mask,dtype=torch.float32)
        output_ids = torch.tensor(output_ids,dtype=torch.long)

        return [input_ids, input_type_ids, input_mask, output_ids, output_mask]

class HLDataset(Dataset):
    def __init__(self, data, tokenizer, label2id, maxlength=503, train=True):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.maxlength = maxlength
        self.train = train
        self.num4label = len(self.label2id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = item['input_ids']
        input_type_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)

        input_ids += [self.tokenizer.pad_token_id] * (self.maxlength - len(input_ids))
        input_type_ids += [self.tokenizer.pad_token_type_id] * (self.maxlength - len(input_type_ids))
        input_mask += [0] * (self.maxlength - len(input_mask))

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_type_ids = torch.tensor(input_type_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)

        inputs_list = [input_ids, input_type_ids, input_mask]

        if self.train:
            labels = np.zeros(self.num4label)
            labels_id = [self.label2id[x] for x in item['labels'] if x in self.label2id]
            labels[labels_id] = 1
            labels = torch.tensor(labels, dtype=torch.float32)
            inputs_list.append(labels)

        return inputs_list

class FTLDataset(Dataset):
    def __init__(self, data, tokenzier, label2id, headlabel2id, maxlength=503, train=True, is_baseline=False):
        self.data = data
        self.tokenizer = tokenzier
        self.label2id = label2id
        self.headlabel2id = headlabel2id
        self.train = train
        self.maxlength = maxlength
        self.num4label = len(label2id)
        self.id2headlabel = {v: k for k, v in self.headlabel2id.items()}
        self.is_baseline = is_baseline
        if self.is_baseline:
            print('Running baseline, no augmentation.')
        else:
            print('Running THETA, using augmentation.')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        pred_labels = item['pred_labels']
        if self.is_baseline:
            # if running baseline, no augmentation
            pred_labels = []
        # pred_labels = []
        headlabels = list(map(lambda x: self.id2headlabel['[unused{}]'.format(x)], pred_labels))

        input_ids = item['input_ids'][1:-1]
        input_ids = input_ids[:self.maxlength - 3 - len(headlabels)]
        input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
        input_type_ids = [0] * len(input_ids)

        if len(pred_labels) > 0:
	        headlabels_token = list(map(lambda x: self.headlabel2id[x], headlabels)) + [self.tokenizer.sep_token]
	        input_ids += self.tokenizer.convert_tokens_to_ids(headlabels_token)
	        input_type_ids += [1] * len(headlabels_token)

        input_mask = [1] * len(input_ids)

        input_ids += [self.tokenizer.pad_token_id] * (self.maxlength - len(input_ids))
        input_type_ids += [self.tokenizer.pad_token_type_id] * (self.maxlength - len(input_type_ids))
        input_mask += [0] * (self.maxlength - len(input_mask))

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_type_ids = torch.tensor(input_type_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)

        inputs_list = [input_ids, input_type_ids, input_mask]

        if self.train:
            labels = np.zeros(self.num4label)
            labels_id = [self.label2id[x] for x in item['labels'] if x in self.label2id]
            labels[labels_id] = 1
            labels = torch.tensor(labels, dtype=torch.float32)
            inputs_list.append(labels)

        return inputs_list