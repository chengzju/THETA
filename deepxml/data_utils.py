# -*- coding: utf-8 -*-
import numpy as np
import json
from tqdm import tqdm
import multiprocessing
from nltk import word_tokenize

def load_data(data_path,label_path,data_output=None,label_output=None):
    data = []
    freq4labels = {}
    with open(data_path, 'r') as f:
        for i, line in enumerate(tqdm(f,desc="Data")):
            item = {'id': i, 'text': line.strip(), 'labels': []}
            data.append(item)
    with open(label_path, 'r') as f:
        for i, line in enumerate(tqdm(f,desc="Label")):
            line_arr = line.strip().split()
            data[i]['labels'] = line_arr
            for x in line_arr:
                freq4labels[x] = freq4labels.get(x, 0) + 1
    if data_output is not None:
        json.dump(data,open(data_output,'w'),indent=4,ensure_ascii=False)
    if label_output is not None:
        json.dump(freq4labels,open(label_output,'w'),indent=4,ensure_ascii=False)
    return data, freq4labels

class MLMDataset(object):
    def __init__(self, tokenizer, label2id, headlabels, mask_rate=0.15, maxlength=500, text_repeat=False):
        self.tokenizer = tokenizer
        self.headlabels = headlabels
        self.label2id = label2id
        self.mask_rate = mask_rate
        self.maxlength = maxlength
        self.text_repeat = text_repeat

    def word_segment(self, text):
        tokens = [word.lower() for word in word_tokenize(text)]
        return tokens

    def token_mask(self, token_id):
        rand = np.random.random()
        if rand <= 0.8:
            return self.tokenizer.mask_token_id
        elif rand <= 0.9:
            return token_id
        else:
            return np.random.randint(0, self.tokenizer.vocab_size)

    def text_mask(self, text):
        words = self.word_segment(text)
        words = words[:self.maxlength * 2 + 1]
        rands = np.random.random(len(words))
        token_ids, token_mask_ids, mask_ids = [], [], []
        for rand, word in zip(rands, words):
            word_tokens = self.tokenizer.tokenize(word)
            word_token_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)
            token_ids.extend(word_token_ids)

            if rand < self.mask_rate:
                mask_tokens = [self.token_mask(x) for x in word_token_ids]
                mask_id = [1] * len(mask_tokens)
            else:
                mask_tokens = word_token_ids
                mask_id = [0] * len(mask_tokens)

            token_mask_ids.extend(mask_tokens)
            mask_ids.extend(mask_id)

        return [token_ids, token_mask_ids, mask_ids]

    def label_mask(self, label_ids, add_label_ids=None):
        if len(label_ids) > 0:
            rands = np.random.random(len(label_ids))
            token_ids, token_mask_ids, mask_ids = [], [], []
            label_ids = self.tokenizer.convert_tokens_to_ids(label_ids)
            for rand, word in zip(rands, label_ids):
                if rand < self.mask_rate:
                    mask_tokens = self.token_mask(word)
                    mask_id = 1
                else:
                    mask_tokens = word
                    mask_id = 0

                token_ids.append(word)
                token_mask_ids.append(mask_tokens)
                mask_ids.append(mask_id)
            if add_label_ids is not None:
                token_ids.extend(add_label_ids)
                token_mask_ids.extend(add_label_ids)
                mask_ids.extend([0] * len(add_label_ids))
            token_ids.append(self.tokenizer.sep_token_id)
            token_mask_ids.append(self.tokenizer.sep_token_id)
            mask_ids.append(0)
        else:
            token_ids, token_mask_ids, mask_ids = [], [], []
        return [token_ids, token_mask_ids, mask_ids]

    def sentence_segment(self, text, pad_num):
        tokens_ids, tokens_mask_ids, mask_ids = text
        seq_length = self.maxlength - pad_num
        tokens_ids_list, tokens_mask_ids_list, mask_ids_list = [], [], []
        one_tokens_ids = tokens_ids[:seq_length]
        one_tokens_mask_ids = tokens_mask_ids[:seq_length]
        one_mask_ids = mask_ids[:seq_length]
        tokens_ids_list.append(one_tokens_ids)
        tokens_mask_ids_list.append(one_tokens_mask_ids)
        mask_ids_list.append(one_mask_ids)
        if len(tokens_ids) > seq_length and self.text_repeat:
            one_tokens_ids = tokens_ids[-seq_length:]
            one_tokens_mask_ids = tokens_mask_ids[-seq_length:]
            one_mask_ids = mask_ids[-seq_length:]
            tokens_ids_list.append(one_tokens_ids)
            tokens_mask_ids_list.append(one_tokens_mask_ids)
            mask_ids_list.append(one_mask_ids)
        return tokens_ids_list, tokens_mask_ids_list, mask_ids_list

    def process(self, data, output=None, add_label2id=None,tid=None,process_num=4):
        new_data = []
        max_token_length = 0
        for i,item in enumerate(tqdm(data,desc="MLM Data Preocess")):
            if tid is not None and i % process_num != tid:continue
            text = item['text']
            labels = list(set(item['labels']) & set(self.headlabels[0]))
            add_labels = list(set(item['labels'])&set(self.headlabels[1]))
            labels_id = [self.label2id[x] for x in labels]
            labels_num = len(labels)
            if add_label2id is not None:
                add_labels_id = [self.label2id[x] for x in add_labels]
                add_labels_id = [add_label2id[x] for x in add_labels_id]
                labels_num += len(add_labels_id)
            else:
                add_labels_id = None
            tokens_ids, tokens_mask_ids, mask_ids = self.text_mask(text)
            pad_num = labels_num + 1
            tokens_ids_list, tokens_mask_ids_list, mask_ids_list = \
                self.sentence_segment([tokens_ids, tokens_mask_ids, mask_ids], pad_num)
            for tokens_ids, tokens_mask_ids, mask_ids in zip(tokens_ids_list, tokens_mask_ids_list, mask_ids_list):
                label_tokens_ids, label_tokens_mask_ids, label_mask_ids = self.label_mask(labels_id, add_labels_id)
                input_ids = [self.tokenizer.cls_token_id]
                input_ids += tokens_mask_ids
                input_ids.append(self.tokenizer.sep_token_id)
                input_ids += label_tokens_mask_ids

                output_mask = [0]
                output_mask += mask_ids
                output_mask.append(0)
                output_mask += label_mask_ids

                output_ids = [self.tokenizer.cls_token_id]
                output_ids += tokens_ids
                output_ids.append(self.tokenizer.sep_token_id)
                output_ids += label_tokens_ids

                output_mask_idx = np.where(np.array(output_mask)==1)[0]
                output_ids = np.array(output_ids)[output_mask_idx]
                output_mask_idx = list([int(x) for x in output_mask_idx])
                output_ids = list([int(x) for x in output_ids])
                if len(output_mask_idx) > 0:
                    d = {'id': item['id'],
                         'input_ids': input_ids,
                         'output_mask_idx': output_mask_idx,
                         'output_ids': output_ids
                         }
                    new_data.append(d)
                    max_token_length = max(max_token_length, len(input_ids))
        print('lm max length %d, max token length %d' % (self.maxlength, max_token_length))
        if output is not None:
            json.dump(new_data, open(output, 'w'), indent=4, ensure_ascii=False)
        return new_data

    def multiprocess(self, data, output=None, add_label2id=None, process_num=4):
        pool = multiprocessing.Pool(processes=process_num)
        multi_res = [pool.apply_async(self.process, (data, None, add_label2id,tid,process_num)) for tid in
                     range(0, process_num)]
        bags_list = [res.get() for res in multi_res]
        new_data = []
        for sub_data in bags_list:
            new_data.extend(sub_data)
        if output is not None:
            json.dump(new_data, open(output, 'w'), indent=4, ensure_ascii=False)
        return new_data

def xml_sentence_segment(data, tokenizer, maxlength=500,output=None,tid=None,process_num=4):
    new_data = []
    max_token_length = 0
    for i, item in enumerate(tqdm(data)):
        if tid is not None and i % process_num != tid: continue
        d = {'id': item['id'], 'labels': item['labels']}
        text = item['text']
        input_ids = tokenizer.tokenize(text)
        input_ids = input_ids[:maxlength]
        d['input_ids'] = [tokenizer.cls_token] + input_ids + [tokenizer.sep_token]
        d['input_ids'] = tokenizer.convert_tokens_to_ids(d['input_ids'])
        max_token_length = max(max_token_length, len(d['input_ids']))
        new_data.append(d)
    print('clf max length %d, max token length %d' % (maxlength, max_token_length))
    if output is not None:
        json.dump(new_data, open(output, 'w'), indent=4, ensure_ascii=False)
    return new_data

def multi_xml_sentence_segment(data,tokenizer,maxlength=500,output=None,process_num=4):
    pool = multiprocessing.Pool(processes=process_num)
    multi_res = [pool.apply_async(xml_sentence_segment,
                                  (data, tokenizer, maxlength, None, tid, process_num)) for tid in range(0, process_num)]
    bags_list = [res.get() for res in multi_res]
    new_data = []
    for sub_data in bags_list:
        new_data.extend(sub_data)
    if output is not None:
        json.dump(new_data, open(output, 'w'), indent=4, ensure_ascii=False)
    return new_data






