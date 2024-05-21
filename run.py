# coding: utf-8
# @Time: 2024/3/26 10:33
# @Author: 从文
# @FileName: run.py
# @Software: PyCharm Community Edition
import argparse
import random
import time

import numpy as np
import torch

from DDRN.datasets.load_data import LoadData
from DDRN.models.DDRN import DDRN
from DDRN.utils.early_stopping import EarlyStopping
from DDRN.utils.evaluate import Evaluator
from DDRN.utils.utils import cim_logger, weights_init

# 配置参数
parser = argparse.ArgumentParser(description="CI")
parser.add_argument("--feature_dim", default=26, type=int)
parser.add_argument("--data_source", default='acic2016', type=str)
parser.add_argument("--data_path", default='datasets/acic2016/', type=str)
parser.add_argument("--seed", default=1234, type=int, help="random seed")
parser.add_argument("--batch_size", default=256, type=int, help='batch_size')
parser.add_argument("--lr", default=1e-2, type=float, help="learning rate")
parser.add_argument("--lrd", default=0.95, type=float, help="learning rate decay")
parser.add_argument("--weight_decay", default=1e-3, type=float, help="L2 regularization parameters")
parser.add_argument("--epochs", default=1500, type=int, help="number of epochs")
parser.add_argument("--p_alpha", default=1.0, type=float, help="treatment risk")
parser.add_argument("--p_beta", default=0.25, type=float, help="MMD loss")
parser.add_argument("--p_lambda", default=0.25, type=float, help="propensity loss",)
parser.add_argument("--p_delta", default=0.25, type=float, help="disentangle loss")
parser.add_argument("--hidden_dim", default=100, type=int, help="hidden layer dim")
parser.add_argument("--expert_dim", default=320, type=int, help="expert hidden layer dim")
parser.add_argument("--num_expert", default=3, type=int, help="number of experts")
parser.add_argument("--num_task", default=3, type=int, help="number of tasks")
parser.add_argument("--num_head", default=4, type=int, help="number of multi-head attentions")

args = parser.parse_args()

# Set logger
logger = cim_logger(f"logs/{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}.log")
logger.info("Begin training & testing causal inference model.")

# Write hyper-parameters into logger
logger.info("Hyper-parameters: %s" % args.__str__())

# Set random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Early stopping
early_stopping = EarlyStopping(patience=50, verbose=True, delta=0.001)

# Load Data
load_data = LoadData(data_src=args.data_source)
raw_data = load_data.read_acic(args.data_path, args.batch_size)
train_data_iterator, test_data_iterator = load_data.split_acic_data(raw_data, args.batch_size)

# Define causal inference model
model = DDRN(feature_dim=train_data_iterator.dataset.x.shape[-1], hidden_dim=args.hidden_dim, expert_dim=args.expert_dim, n_expert=args.num_expert, n_task=args.num_task, n_head=args.num_head, use_gates=True)
model.apply(weights_init)

opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=args.lrd) # default step size is 20

# Training
step = 0
best_eval = {"best_out-of_pehe": 9.9, "best_out-of_policy_risk": 9.9}
for epoch in range(args.epochs):
    model.train()
    for t, yf, ycf, mu0, mu1, x in train_data_iterator:
        step += 1
        _, _, y_pred, reg_loss, treat_bce_loss, mmd_loss, prop_loss, dis_loss = model(x, t, yf, ycf)
        loss = reg_loss + args.p_alpha * treat_bce_loss + args.p_beta * mmd_loss + args.p_lambda * prop_loss + args.p_delta * dis_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=6.0)
        opt.step()

    if epoch % 10 == 0:
        with torch.no_grad():
            # Predict test data
            y1pred_test, y0pred_test, y_hat_test, reg_loss_test, treat_bce_loss_test, mmd_loss_test, prop_loss_test, dis_loss_test = model(
                x=test_data_iterator.dataset.x,
                t=test_data_iterator.dataset.t,
                yf=test_data_iterator.dataset.yf,
                ycf=test_data_iterator.dataset.ycf)
            test_loss = reg_loss_test + args.p_alpha * treat_bce_loss_test + args.p_beta * mmd_loss_test + args.p_lambda * prop_loss_test + args.p_delta * dis_loss_test
            evaluator_test = Evaluator(t=test_data_iterator.dataset.t,
                                       yf=test_data_iterator.dataset.yf,
                                       ycf=test_data_iterator.dataset.ycf,
                                       mu0=test_data_iterator.dataset.mu0,
                                       mu1=test_data_iterator.dataset.mu1)
            ate_test, pehe_test, within_pehe_test = evaluator_test.calculate_stats(y1pred=y1pred_test, y0pred=y0pred_test)
            best_eval['outof_pehe'] = pehe_test.item()
            best_eval['outof_ate'] = ate_test.item()

            # Update best evaluation result
            if pehe_test < best_eval['best_out-of_pehe']:
                best_eval['best_out-of_pehe'] = pehe_test.item()
                best_eval['best_epochs'] = epoch
            # Print logger
            logger.info(
                f"Epochs: {epoch}, learning rate: {scheduler.get_last_lr()}, outof loss: {'%.5f' % test_loss}, "
                f"best out-of pehe: {'%.5f' % best_eval['best_out-of_pehe']}, out-of pehe: {'%.5f' % best_eval['out-of_pehe']}, within pehe: {'%.5f' % best_eval['within_pehe']}, out-of ate: {'%.5f' % best_eval['out-of_ate']}, within ate: {'%.5f' % best_eval['within_ate']}")
            # Determine if the conditions are met.
            early_stopping(val_loss=best_eval['best_out-of_pehe'], model=model)
            if early_stopping.early_stop:
                print('Early stop model training process.')
                break
    epoch += 1
    scheduler.step()

# model.writer.close()
logger.info(f"In epochs: {best_eval['best_epochs']}, we get best test dataset pehe: {best_eval['best_outof_pehe']}.")