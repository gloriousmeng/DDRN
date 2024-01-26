# coding: utf-8
# @Time: 2024/1/26 14:18
# @Author: **
# @FileName: evaluate.py
# @Software: PyCharm Community Edition
import numpy as np
import torch
from sklift.metrics import uplift_auc_score


class Evaluator(object):
    def __init__(self, t, yf, ycf=None, mu1=None, mu0=None):
        self.y = yf
        self.t = t
        self.y_cf = ycf
        self.mu0 = mu0
        self.mu1 = mu1
        if mu0 is not None and mu1 is not None:
            self.true_ite = mu1 - mu0

    def auuc(self, y1_pred, y0_pred):
        uplift_score = y1_pred - y0_pred
        auuc = uplift_auc_score(y_true=self.y, uplift=uplift_score, treatment=self.t)
        return auuc

    def pehe(self, y1pred, y0pred):
        return torch.sqrt(torch.mean(torch.square((self.mu1 - self.mu0) - (y1pred - y0pred))))

    def abs_ate(self, y1pred, y0pred):
        ypred1 = y1pred.squeeze(1)
        ypred0 = y0pred.squeeze(1)
        return torch.abs(torch.mean(ypred1 - ypred0) - torch.mean(self.true_ite))

    def within_pehe(self, ypred1, ypred0):
        i0 = torch.where(self.t < 1)[0]
        i1 = torch.where(self.t > 0)[0]

        y1_true = torch.index_select(self.y, dim=0, index=i1)
        y0_true = torch.index_select(self.y, dim=0, index=i0)

        y1_pred = torch.index_select(ypred1, dim=0, index=i0)
        y0_pred = torch.index_select(ypred0, dim=0, index=i1)

        y1 = torch.cat([y1_true, y1_pred], dim=0)
        y0 = torch.cat([y0_pred, y0_true], dim=0)

        return torch.sqrt(torch.mean(torch.square((self.mu1 - self.mu0) - (y1 - y0))))

    def policy_risk(self, y1pred, y0pred):
        policy = ((y1pred - y0pred) > 0)

        treat_overlap = (policy == self.t) * (self.t > 0)
        control_overlap = (policy == self.t) * (self.t < 1)

        treat_value = 0 if torch.sum(treat_overlap) == 0 else torch.mean(self.y[treat_overlap])
        control_value = 0 if torch.sum(control_overlap) == 0 else torch.mean(self.y[control_overlap])

        pit = np.mean(policy.detach().numpy())
        policy_value = pit * treat_value + (1 - pit) * control_value
        policy_risk = 1 - policy_value

        return policy_risk

    def att_bias(self, y1pred, y0pred):
        # Calculate factual att
        i0 = torch.where(self.t < 1)[0]
        i1 = torch.where(self.t > 0)[0]
        y_treat = torch.index_select(self.y, dim=0, index=i1)
        y_control = torch.index_select(self.y, dim=0, index=i0)
        att = torch.abs(torch.mean(y_treat) - torch.mean(y_control))
        # Calculate predict att
        pred_att = torch.abs(torch.mean(torch.index_select(y1pred, dim=0, index=i1)) - torch.mean(torch.index_select(y0pred, dim=0, index=i1)))

        att_bias = torch.abs(att - pred_att)
        return att_bias

    def calculate_stats(self, y1pred, y0pred):
        ate = self.abs_ate(y1pred, y0pred)
        pehe = self.pehe(y1pred, y0pred)
        within_pehe = self.within_pehe(y1pred, y0pred)

        return ate, pehe, within_pehe

    def without_cf_calculate_stats(self, y1pred, y0pred):
        att_bias = self.att_bias(y1pred=y1pred, y0pred=y0pred)
        policy_risk = self.policy_risk(y1pred=y1pred, y0pred=y0pred)

        return att_bias, policy_risk

