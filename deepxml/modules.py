# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from collections import OrderedDict
from future.utils import iteritems

def convert_weights(state_dict):
    tmp_weights = OrderedDict()
    for name, params in iteritems(state_dict):
        tmp_weights[name.replace('module.', '')] = params
    return tmp_weights

def load_model(model, model_path):
    try:
        model.load_state_dict(torch.load(model_path))
    except:
        model.load_state_dict(convert_weights(torch.load(model_path)))
    return model

def mlm_loss(y_pred, y_true, mask, eps=1e-10):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    y_pred = y_pred.contiguous().view(-1, y_pred.shape[-1])
    y_true = y_true.contiguous().view(-1)
    mask = mask.contiguous().view(-1)
    loss = loss_fct(y_pred, y_true)
    loss = torch.sum(loss * mask) / (torch.sum(mask) + eps)
    return loss

def xml_loss(y_pred,y_true):
    loss_fct = nn.BCEWithLogitsLoss()
    loss = loss_fct(y_pred, y_true)
    return loss