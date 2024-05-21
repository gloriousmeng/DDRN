# coding: utf-8
# @Time: 2024/3/26 10:33
# @Author: 从文
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


class TreatDense(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, output_dim):
        super(TreatDense, self).__init__()
        self.fc_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        fc_out = self.fc_net(x)
        return fc_out


class PropensityDense(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, output_dim):
        super(PropensityDense, self).__init__()
        self.prop_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, int(hidden_dim / 2)),
            nn.BatchNorm1d(int(hidden_dim / 2)),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(int(hidden_dim / 2), output_dim),
        )

    def forward(self, x):
        prop_net = self.prop_net(x)
        return prop_net


class AdjustDense(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, output_dim):
        super(AdjustDense, self).__init__()
        self.fc_net_treat = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.fc_net_control = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs):
        x, t = inputs
        # Splitting treatment
        i0 = torch.where(t < 1)[0]
        i1 = torch.where(t > 0)[0]
        # Splitting output by treatment
        y1pred = self.fc_net_treat(x)
        y0pred = self.fc_net_control(x)

        # Interleave two tensors
        _indices = torch.argsort(torch.cat([i0, i1], dim=0))
        y_hat = torch.cat([y0pred[i0], y1pred[i1]], dim=0)[_indices]
        return y1pred, y0pred, y_hat


class DDRN(nn.Module):
    def __init__(self, feature_dim, hidden_dim, expert_dim, n_expert, n_task, n_head, use_gates=False, log_dir='runs'):
        super(DDRN, self).__init__()
        self.feature_dim = feature_dim
        self.expert_qdim = expert_dim

        # Define expert multi-head attention
        self.expert_attention = Expert_Attention(feature_dim=feature_dim, expert_dim=expert_dim, n_expert=n_expert, n_task=n_task, num_heads=n_head, use_att=use_gates)

        # Define treatment tower
        self.treatment_tower = nn.Sequential(
            TreatDense(input_dim=expert_dim * 2, hidden_dim=hidden_dim, dropout=0.4, output_dim=1),
        )

        # Define adjustment tower
        self.adjustment_tower = nn.Sequential(
            AdjustDense(input_dim=expert_dim * 2, hidden_dim=hidden_dim, dropout=0.4, output_dim=1),
        )

        # Define propensity score tower
        self.propensity_tower = nn.Sequential(
            PropensityDense(input_dim=expert_dim, hidden_dim=hidden_dim, dropout=0.4, output_dim=1)
        )

        # Define affine network
        self.treat_affine_network = nn.Sequential(
            nn.Linear(self.expert_dim, self.expert_dim),
            nn.LayerNorm(self.expert_dim),
        )
        self.target_affine_network = nn.Sequential(
            nn.Linear(self.expert_dim, self.expert_dim),
            nn.LayerNorm(self.expert_dim),
        )
        self.confound_affine_network = nn.Sequential(
            nn.Linear(self.expert_dim, self.expert_dim),
            nn.LayerNorm(self.expert_dim),
        )

        # Define sparse vector
        self.wt = nn.Parameter(torch.rand(self.expert_dim), requires_grad=True)
        self.wc = nn.Parameter(torch.rand(self.expert_dim), requires_grad=True)
        self.wa = nn.Parameter(torch.rand(self.expert_dim), requires_grad=True)

        # self.writer = SummaryWriter(log_dir=log_dir, flush_secs=120)

    def forward(self, x, t, yf, ycf=None):
        # Three representation towers
        treatment_expert, confound_expert, adjustment_expert = self.expert_attention(x=x.float())
        # Disentangle representation
        treatment_rest = self.treat_affine_network(torch.multiply(treatment_expert, self.wt))
        confound_rest = self.confound_affine_network(torch.multiply(confound_expert, self.wc))
        adjustment_rest = self.target_affine_network(torch.multiply(adjustment_expert, self.wa))

        # Calculate propensity score & propensity loss
        treatment_logits = self.treatment_tower(torch.cat([treatment_rest, confound_rest], dim=-1))
        propensity_logits = self.propensity_tower(confound_rest)

        # Calculate treatment binary cross entropy loss
        treatment_loss = torch.mean(torch.binary_cross_entropy_with_logits(treatment_logits, t))

        # Calculate treatment imbalance loss
        treat_imb_mmd = self.treatment_rectification(treatment_rest, yf, ycf)

        # Calculate sample weight
        sample_weight = self.sample_weight(t, pi0=self.propensity_score(propensity_logits, t), is_ipw=True)

        # Calculate propensity loss
        propensity_loss = torch.mean(torch.binary_cross_entropy_with_logits(propensity_logits, t))

        # Regression logits
        y1pred, y0pred, y_hat = self.adjustment_tower([torch.cat([adjustment_rest, confound_rest], dim=-1), t])
        # Calculate regression imbalance loss
        adjust_imb_mmd = self.mmd_lin(adjustment_rest, t)

        # Calculate adjustment square loss
        regression_loss = torch.mean(F.mse_loss(y_hat, yf, reduction='none') * sample_weight)
        # Calculate imbalance mmd loss
        imb_mmd = adjust_imb_mmd + treat_imb_mmd

        # Calculate disentangle loss
        disentangle_loss = torch.sqrt(torch.square(torch.sum(self.wt * self.wc) + torch.sum(self.wt * self.wa) + torch.sum(self.wc * self.wa)))

        return y1pred, y0pred, y_hat, regression_loss, treatment_loss, imb_mmd, propensity_loss, disentangle_loss

    @staticmethod
    def propensity_loss(prop_logits, t):
        sigma = torch.sigmoid(prop_logits)
        prop_loss = - torch.mean(t * torch.log(sigma + 1e-4) + (1 - t) * torch.log(1 - sigma + 1e-4))
        return prop_loss

    @staticmethod
    def treatment_rectification(treat_rest, yf, ycf=None):
        # Calculate y median value
        y_mid = torch.mean(torch.cat([yf, ycf], dim=1), dim=1, keepdim=True) if ycf is not None else torch.mean(yf)

        # Calculate great/less idx
        idx_great = torch.where(yf.__ge__(y_mid))[0]
        idx_less = torch.where(yf.__lt__(y_mid))[0]
        pt = len(idx_great) / (len(idx_great) + len(idx_less))

        # Splitting treatment representation by treatment
        mean_great = torch.mean(treat_rest[idx_great], dim=0)
        mean_less = torch.mean(treat_rest[idx_less], dim=0)
        treat_mmd_lin = torch.sum(torch.square(2.0 * pt * mean_great - 2.0 * (1.0 - pt) * mean_less))

        return treat_mmd_lin

    @staticmethod
    def sparse_loss(sl):
        sparse_loss = torch.mean(torch.abs(1.0 - 2.0 * torch.abs(sl - 0.5)))
        return sparse_loss

    @staticmethod
    def mmd_lin(x_rest, t):
        # Splitting treatment index
        c_idx = torch.where(t < 1)[0]
        t_idx = torch.where(t > 0)[0]
        p = len(t_idx) / (len(t_idx) + len(c_idx))

        # Splitting x representation by treatment
        mean_treated = torch.mean(x_rest[t_idx], dim=0)
        mean_control = torch.mean(x_rest[c_idx], dim=0)

        mmd = torch.sum(torch.square(2.0 * p * mean_treated - 2.0 * (1.0 - p) * mean_control))

        return mmd

    @staticmethod
    def sample_weight(t, pi0=None, is_ipw=False):
        p_t = np.mean(t.cpu().detach().numpy())
        w_t = t / (2 * p_t)
        w_c = (1 - t) / (2 * (1 - p_t))
        if p_t > 0.0:
            if is_ipw and pi0 is not None:
                sample_weight = (1 + (1 - pi0) / pi0 * (p_t / (1 - p_t))**(2.0 * t - 1.0)) * np.sqrt(w_t + w_c)
            else:
                sample_weight = w_t + w_c
        else:
            sample_weight = torch.ones((len(t), 1))
        sample_weight = torch.clamp(sample_weight, 0.0, 3.0)
        return sample_weight

    @staticmethod
    def propensity_score(propensity, t):
        sigma = torch.sigmoid(propensity)
        pi0 = torch.multiply(t, sigma) + torch.multiply(1 - t, 1 - sigma)
        return pi0

    def save_to_local(self, steps):
        model_filename = f"causal_inference_{steps}_{datetime.now().strftime('%Y%m%dT%H%M%S')}.pth"
        saved_model_dir = os.path.abspath(os.curdir) + '/saved_models/'
        if not os.path.exists(saved_model_dir):
            os.makedirs(saved_model_dir)
        # Get oldest checkpoint
        get_oldest_file(saved_model_dir)
        # save model
        torch.save(self.state_dict(), os.path.join(saved_model_dir, model_filename))

        print(f"At steps: {steps}, saving the model weights to local: { os.path.join(saved_model_dir, model_filename)}.")
