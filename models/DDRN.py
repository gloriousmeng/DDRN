# coding: utf-8
# @Time: 2024/1/26 10:33
# @Author: **
# @FileName: DDRN.py
# @Software: PyCharm Community Edition

import os
from datetime import datetime

import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
# from torch.utils.tensorboard import SummaryWriter

from DDRN.layers.attention import MultiHeadAttention
from DDRN.utils.utils import get_oldest_file
from sklift.metrics import uplift_auc_score, qini_auc_score


class Expert(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=100, dropout=0.0):
        super(Expert, self).__init__()
        self.expert_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        # T/A/C expert represent network
        expert_net = self.expert_net(x)
        return expert_net


class Expert_Attention(nn.Module):
    def __init__(self, feature_dim, expert_dim, n_task, n_expert, num_heads=2, dropout=0.5, use_att=True):
        super(Expert_Attention, self).__init__()
        self.n_task = n_task
        self.n_expert = n_expert
        self.feature_dim = feature_dim
        self.expert_dim = expert_dim
        self.use_att = use_att
        self.num_heads = num_heads
        self.ffn = nn.Linear(expert_dim, expert_dim)
        if use_att:
            self.attention = MultiHeadAttention(d_model=expert_dim, num_heads=self.num_heads)

        for i in range(n_expert):
            setattr(self, 'expert_layer' + str(i + 1), Expert(feature_dim, expert_dim, dropout=dropout))
        self.expert_layers = [getattr(self, 'expert_layer' + str(i + 1)) for i in range(n_expert)]

    def forward(self, x):
        if self.use_att is False:
            towers = [ex(x) for ex in self.expert_layers]
        else:
            expert_nets = [ex(x) for ex in self.expert_layers]
            stacked_experts = torch.stack(expert_nets, dim=1)  # shape: [bs, n_exp, hidden_dim]
            att_expert_nets = self.attention.forward(stacked_experts, stacked_experts, stacked_experts)
            towers = [att_expert_nets[:, i, :] for i in range(self.n_task)]

        return towers


