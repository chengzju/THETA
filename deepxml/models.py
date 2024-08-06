# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from  tqdm import tqdm
# from apex import amp
from scipy.stats import logistic
import time

from deepxml.metric import *
from deepxml.modules import *

class Model(object):

    def __init__(self,args):
        self.args = args
        self.state = {}

    def log_write(self, log_str):
        if not self.args.local_rank == 0:
            return
        print(log_str)
        if self.args.log_path is not None and os.path.exists(self.args.log_path):
            with open(self.args.log_path, 'a+') as writer:
                writer.write(log_str)

    def save_model(self, model, model_path):
        torch.save(model.state_dict(), model_path)

    def load_model(self, model, model_path):
        return load_model(model,model_path)

    def swa_init(self,model_list):
        if 'swa_list' not in self.state:
            log_str = 'SWA Initializing'
            self.log_write(log_str)
            swa_state_list = []
            for model in model_list:
                swa_state = {'models_num': 1}
                for n, p in model.named_parameters():
                    swa_state[n] = p.data.clone().detach()
                swa_state_list.append(swa_state)
            self.state['swa_list'] = swa_state_list

    def swa_step(self, model_list):
        if 'swa_list' in self.state:
            swa_state_list = self.state['swa_list']
            for i, swa_state in enumerate(swa_state_list):
                swa_state['models_num'] += 1
                beta = 1.0 / swa_state['models_num']
                model = model_list[i]
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        swa_state[n].mul_(1.0 - beta).add_(beta, p.data)

    def swap_swa_params(self, model_list):
        if 'swa_list' in self.state:
            swa_state_list = self.state['swa_list']
            for i, swa_state in enumerate(swa_state_list):
                model = model_list[i]
                for n, p in model.named_parameters():
                    p.data, swa_state[n] = swa_state[n], p.data

    def disable_swa(self):
        if 'swa_list' in self.state:
            del self.state['swa_list']

    def tqdm_wrapper(self,obj):
        if self.args.local_rank == 0:
            return tqdm(obj)
        else:
            return obj

class MLMModel(Model):
    def __init__(self, args):
        super(MLMModel, self).__init__(args)

    def train(self, encoder, network, optimizer, train_loader, valid_loader, device):
        args = self.args
        model_type = args.model_type
        save_path = os.path.join(args.save_path, model_type)
        if not os.path.isdir(save_path) and self.args.local_rank == 0:
            os.makedirs(save_path)
        elif not os.path.isdir(save_path) and not self.args.local_rank == 0:
            time.sleep(12)
        log_path = os.path.join(save_path, 'train.log')
        args.log_path = log_path
        log_str = 'model type: {}'.format(model_type)
        if self.args.local_rank == 0:
            with open(log_path, 'w') as writer:
                writer.write(log_str)
        log_str = 'load data from {}'.format(args.data_dir)
        self.log_write(log_str)

        best_score = -1
        epochs_without_imp = 0
        best_epoch = 0

        model_list = [encoder, network]

        for epoch in range(1,args.epochs+1):
            if epoch == args.swa_warmup:
                self.swa_init(model_list)
            log_str = '\nEpoch: %d/%d ' % (epoch, args.epochs)
            self.log_write(log_str)
            encoder.train()
            network.train()
            log_str = 'encoder lr: {}, clf model lr: {}, gpu id: {}'.format(optimizer.param_groups[0]['lr'],
                                                                            optimizer.param_groups[1]['lr'],
                                                                            args.gpuid)
            self.log_write(log_str)
            for batch_idx, inputs_list in enumerate(self.tqdm_wrapper(train_loader)):
                input_ids, input_type_ids, input_mask, output_ids, output_mask = inputs_list
                input_ids, input_type_ids, input_mask, output_ids, output_mask = \
                    input_ids.to(device), input_type_ids.to(device), input_mask.to(device),\
                    output_ids.to(device), output_mask.to(device)
                last_hidden_state, cls = encoder(input_ids=input_ids,
                                                 attention_mask=input_mask,
                                                  token_type_ids=input_type_ids)
                inputs = [last_hidden_state, cls]
                logits = network.forward(inputs)
                loss = mlm_loss(logits, output_ids, output_mask) + torch.sum(cls) * 0
                optimizer.zero_grad()
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()
            self.swa_step(model_list)
            self.swap_swa_params(model_list)
            outputs = self.predict(encoder, network, valid_loader, device)
            score = self.eval(outputs[0], outputs[1])

            if args.local_rank == 0:
                if score > best_score:
                    self.save_model(encoder, save_path + "/BEST_encoder_checkpoint.pt")
                    self.save_model(network, save_path + "/BEST_mlm_checkpoint.pt")
                    log_str = "\nNow the best score is %.6f, it was %.6f\n" % (score, best_score)
                    # print(log_str)
                    self.log_write(log_str)
                    best_score = score
                    best_epoch = epoch
                    epochs_without_imp = 0
                else:
                    epochs_without_imp += 1
                    log_str = "\nBest score is still %.6f," \
                              "best_epoch is %d," \
                              "epochs without imp. %d\n" % (best_score, best_epoch, epochs_without_imp)
                    # print(log_str)
                    self.log_write(log_str)

                    if epoch - best_epoch > args.patience:
                        print("Early stopping")
                        return
            self.swap_swa_params(model_list)

    def predict(self, encoder, network, valid_loader, device):
        encoder.eval()
        network.eval()
        outputs = [[], []] #label, pred
        with torch.no_grad():
            for batch_idx, inputs_list in enumerate(self.tqdm_wrapper(valid_loader)):
                input_ids, input_type_ids, input_mask, output_ids, output_mask = inputs_list
                input_ids, input_type_ids, input_mask, output_ids, output_mask = \
                    input_ids.to(device), input_type_ids.to(device), input_mask.to(device), \
                    output_ids.to(device), output_mask.to(device)
                last_hidden_state, cls = encoder(input_ids=input_ids,
                                                 attention_mask=input_mask,
                                                 token_type_ids=input_type_ids)
                inputs = [last_hidden_state, cls]
                logits = network.forward(inputs)
                _, pred = torch.max(logits, dim=2)
                pred = pred[output_mask > 0]
                output_ids = output_ids[output_mask > 0]
                outputs[0].append(output_ids.data.cpu().numpy())
                outputs[1].append(pred.data.cpu().numpy())
        outputs[0] = np.concatenate(outputs[0], axis=0)
        outputs[1] = np.concatenate(outputs[1], axis=0)
        return outputs

    def eval(self, y_true, y_pred):
        log_str = '\nmlm metric'
        self.log_write(log_str)
        acc = mlm_metric(y_true, y_pred)
        log_str = '\nacc: %.6f'%(acc)
        self.log_write(log_str)
        return acc

class XMLModel(Model):
    def __init__(self, args):
        super(XMLModel, self).__init__(args)

    def train(self, encoder, network, optimizer, train_loader, valid_loader, device):
        args = self.args
        model_type = args.model_type
        save_path = os.path.join(args.save_path, model_type)
        if not os.path.isdir(save_path) and self.args.local_rank == 0:
            os.makedirs(save_path)
        elif not os.path.isdir(save_path) and not self.args.local_rank == 0:
            time.sleep(12)
        log_path = os.path.join(save_path, 'train.log')
        args.log_path = log_path
        log_str = 'model type: {}'.format(model_type)
        self.log_write(log_str)
        log_str = 'load data from {}'.format(args.data_dir)
        self.log_write(log_str)

        if self.args.local_rank == 0:
            with open(log_path, 'w') as writer:
                writer.write(log_str)

        best_score = -1
        epochs_without_imp = 0
        best_epoch = 0
        model_list = [encoder, network]
        for epoch in range(1,args.epochs+1):
            if epoch == args.swa_warmup:
                self.swa_init(model_list)
            log_str = '\nEpoch: %d/%d ' % (epoch, args.epochs)
            self.log_write(log_str)
            encoder.train()
            network.train()
            log_str = 'encoder lr: {}, clf model lr: {}, gpu id: {}'.format(optimizer.param_groups[0]['lr'],
                                                                        optimizer.param_groups[1]['lr'],
                                                                        args.gpuid)
            self.log_write(log_str)
            for batch_idx, inputs_list in enumerate(self.tqdm_wrapper(train_loader)):
                input_ids, input_type_ids, input_mask, labels = \
                    inputs_list[0], inputs_list[1], inputs_list[2], inputs_list[3]
                input_ids, input_type_ids, input_mask, labels = \
                    input_ids.to(device), input_type_ids.to(device), input_mask.to(device), labels.to(device)
                last_hidden_state, cls = encoder(input_ids=input_ids,
                                                 attention_mask=input_mask,
                                                 token_type_ids=input_type_ids)
                inputs = [last_hidden_state, cls, input_mask]
                logits = network.forward(inputs)
                loss = xml_loss(logits, labels) + torch.sum(cls) * 0
                optimizer.zero_grad()
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()
            self.swa_step(model_list)
            self.swap_swa_params(model_list)
            if not self.args.test_model:
                score = self.predict(encoder, network, valid_loader, device)
            else:
                outputs = self.predict(encoder, network, valid_loader, device)
                score = self.eval(outputs[0], outputs[1])
            
            if args.local_rank == 0:
                if score > best_score:
                    self.save_model(encoder, save_path + "/BEST_encoder_checkpoint.pt")
                    self.save_model(network, save_path + "/BEST_xml_checkpoint.pt")
                    log_str = "\nNow the best score is %.6f, it was %.6f\n" % (score, best_score)
                    self.log_write(log_str)
                    best_score = score
                    best_epoch = epoch
                    epochs_without_imp = 0
                else:
                    epochs_without_imp += 1
                    log_str = "\nBest score is still %.6f, best_epoch is %d,epochs without imp. %d\n" % (
                        best_score,
                        best_epoch,
                        epochs_without_imp)
                    self.log_write(log_str)
                    if epoch - best_epoch > args.patience:
                        print("Early stopping")
                        return
            self.swap_swa_params(model_list)

    def predict(self, encoder, network, valid_loader, device, batch_pred=False, step2eval=False):
        score = 0
        metrics = np.zeros(5)
        tail_metrics = np.zeros(5)
        cnt = 0
        test_num = 0

        encoder.eval()
        network.eval()
        outputs = [[], []]#[labels, logits]
        with torch.no_grad():
            for inputs_list in self.tqdm_wrapper(valid_loader):
                input_ids, input_type_ids, input_mask, labels = \
                    inputs_list[0], inputs_list[1], inputs_list[2], inputs_list[3]
                input_ids, input_type_ids, input_mask, labels = \
                    input_ids.to(device), input_type_ids.to(device), input_mask.to(device), labels.to(device)
                last_hidden_state, cls = encoder(input_ids=input_ids,
                                                 attention_mask=input_mask,
                                                 token_type_ids=input_type_ids)
                inputs = [last_hidden_state, cls, input_mask]
                logits = network.forward(inputs)

                if not self.args.test_model or step2eval == True:
                    labels = labels.data.cpu().numpy()
                    logits = logits.data.cpu().numpy()
                    metrics_iter = self.eval(labels, logits, ret_metrics=True)
                    metrics += metrics_iter * labels.shape[0]

                    tail_metrics_iter = self.eval(labels[:, self.args.head_label_num:], logits[:, self.args.head_label_num:], ret_metrics=True)
                    cnt += np.sum(np.sum(labels[:, self.args.head_label_num:], axis=1) > 0)
                    test_num += labels.shape[0]
                    tail_metrics += tail_metrics_iter * labels.shape[0]

                else:
                    outputs[0].append(labels.data.cpu().numpy())
                    outputs[1].append(logits.data.cpu().numpy().astype(np.float16))

        if not self.args.test_model or step2eval == True:
            metrics = metrics / test_num
            tail_metrics = tail_metrics / cnt
            print('Non-zero Number: ', test_num, cnt)

            log_str = '\nstep2 metric'
            self.log_write(log_str)
            log_str = '\n' + '\t'.join(['p1', 'p3', 'p5', 'n3', 'n5'])
            self.log_write(log_str)
            log_str = '\n' + '\t'.join(['%.6f'] * 5)
            log_str = log_str % tuple(metrics)
            self.log_write(log_str)

            log_str = '\nstep2 metric - tail labels'
            self.log_write(log_str)
            log_str = '\n' + '\t'.join(['p1', 'p3', 'p5', 'n3', 'n5'])
            self.log_write(log_str)
            log_str = '\n' + '\t'.join(['%.6f'] * 5)
            log_str = log_str % tuple(tail_metrics)
            self.log_write(log_str)

            return metrics[4]
        else:
            outputs[0] = np.concatenate(outputs[0], axis=0)
            outputs[1] = np.concatenate(outputs[1], axis=0)
        return outputs

    def eval(self, labels, logits, ret_metrics=False):
        scores = None
        if ret_metrics:
            scores = xml_metric(labels, logits, np.arange(labels.shape[1]), prt_sample_weight=False)
            return np.array(scores)
        else:
            scores = xml_metric(labels, logits, np.arange(labels.shape[1]))

        p1, p3, p5, n3, n5 = scores
        log_str = '\nPrec@1\tPrec@3\tPrec@5\tnDCG@3\tnDCG@5'
        self.log_write(log_str)
        log_str = '\n%.6f\t%.6f\t%.6f\t%.6f\t%.6f' % (p1, p3, p5, n3, n5)
        self.log_write(log_str)
        score = n5
        
        if labels.shape[1] > self.args.head_label_num:
            scores = xml_metric(labels[:, self.args.head_label_num:], 
                logits[:, self.args.head_label_num:], np.arange(labels.shape[1] - self.args.head_label_num))
            p1, p3, p5, n3, n5 = scores
            log_str = '\nTail Prec@1\tPrec@3\tPrec@5\tnDCG@3\tnDCG@5, tail num {}, unzero cnt {}'.format(str(labels.shape[1] - self.args.head_label_num), 
                np.sum((np.sum(labels[:, self.args.head_label_num:], axis=1)>0)*1))
            self.log_write(log_str)
            log_str = '\n%.6f\t%.6f\t%.6f\t%.6f\t%.6f' % (p1, p3, p5, n3, n5)
            self.log_write(log_str)
        
        return score


    def eval_psp(self, labels, logits, train_labels):
        scores = xml_metric_psp(labels, logits, np.arange(labels.shape[1]), train_labels)

        psp1, psp3, psp5 = scores
        log_str = '\nPSP@1\tPSP@3\tPSP@5'
        print(log_str)
        log_str = '\n%.6f\t%.6f\t%.6f' % (psp1, psp3, psp5)
        print(log_str)

class DataStatistics(Model):
    def __init__(self, args):
        super(DataStatistics, self).__init__(args)

    def train(self, encoder, network, optimizer, train_loader, valid_loader, device):
        args = self.args
        model_type = args.model_type
        save_path = os.path.join(args.save_path, model_type)
        log_path = os.path.join(save_path, 'train.log')
        args.log_path = log_path

        best_score = -1
        epochs_without_imp = 0
        best_epoch = 0
        model_list = [encoder, network]
        cnt = 0
        total_cnt = 0
        for batch_idx, inputs_list in enumerate(self.tqdm_wrapper(train_loader)):
            input_ids, input_type_ids, input_mask, labels = \
                inputs_list[0], inputs_list[1], inputs_list[2], inputs_list[3]
            cnt += torch.sum(labels[:, -1] > 0) 
            total_cnt += labels.shape[0]
        print('Train labels number:', cnt, total_cnt)
        
        cnt = 0
        total_cnt = 0
        for batch_idx, inputs_list in enumerate(self.tqdm_wrapper(valid_loader)):
            input_ids, input_type_ids, input_mask, labels = \
                inputs_list[0], inputs_list[1], inputs_list[2], inputs_list[3]
            cnt += torch.sum(torch.sum(labels, dim=1) > 0) 
            total_cnt += labels.shape[0]
        print('Test labels number:', cnt, total_cnt)
