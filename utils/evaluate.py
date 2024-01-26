# coding: utf-8
# @Time: 2024/1/26 14:18
# @Author: **
# @FileName: evaluate.py
# @Software: PyCharm Community Edition
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