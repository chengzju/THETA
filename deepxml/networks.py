# -*- coding: utf-8 -*-
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_bert import BertOnlyMLMHead
from deepxml.modules import *

class MLAttention(nn.Module):
    def __init__(self, labels_num, hidden_size):
        super(MLAttention, self).__init__()
        self.attention = nn.Linear(hidden_size, labels_num, bias=False)
        nn.init.xavier_uniform_(self.attention.weight)

    def forward(self, inputs, masks):
        masks = torch.unsqueeze(masks, 1)  # N, 1, L
        attention = self.attention(inputs).transpose(1, 2).masked_fill(masks, -np.inf)  # N, labels_num, L
        attention = F.softmax(attention, -1)
        return attention @ inputs   # N, labels_num, hidden_size

class MLLinear(nn.Module):
    def __init__(self, linear_size, output_size):
        super(MLLinear, self).__init__()
        self.linear = nn.ModuleList(nn.Linear(in_s, out_s)
                                    for in_s, out_s in zip(linear_size[:-1], linear_size[1:]))
        for linear in self.linear:
            nn.init.xavier_uniform_(linear.weight)
        self.output = nn.Linear(linear_size[-1], output_size)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, inputs):
        linear_out = inputs
        for linear in self.linear:
            linear_out = F.relu(linear(linear_out))
        return torch.squeeze(self.output(linear_out), -1)

class MLMNetwork(nn.Module):
    def __init__(self,bert_config):
        super(MLMNetwork, self).__init__()
        self.bert_config = bert_config
        self.fc4lm = BertOnlyMLMHead(self.bert_config)

    def forward(self, inputs):
        last_hidden_state, cls = inputs
        logits = self.fc4lm(last_hidden_state)  # [batch_size,seq_len,vocab_size]
        return logits

class LWANetwork(nn.Module):
    def __init__(self, bert_config, num4label):
        super(LWANetwork, self).__init__()
        hidden_size = bert_config.hidden_size
        self.attention = MLAttention(num4label, hidden_size)
        linear_size = [hidden_size]
        self.fc = MLLinear([hidden_size] + linear_size, 1)

    def forward(self, inputs):
        last_hidden_state, cls = inputs[0], inputs[1]
        masks = inputs[2]
        attn_out = self.attention(last_hidden_state, (1 - masks).type(torch.bool))
        logits = self.fc(attn_out)
        return logits




