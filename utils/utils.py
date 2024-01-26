# coding: utf-8
# @Time: 2024/1/26 11:38
# @Author: **
# @FileName: utils.py
# @Software: PyCharm Community Edition
import logging
import os

import pandas as pd
from torch import nn


def get_oldest_file(directory, threshold=10):
    # Initial variable
    oldest = None
    # Find the oldest file
    for file in os.listdir(directory):
        # Set path
        path = os.path.join(directory, file)
        # Logistic judge
        if not os.path.isdir(path) and '.py' not in file:
            modified_time = os.stat(path).st_mtime
            if oldest is None or modified_time < oldest[1]:
                oldest = (path, modified_time)
    # Delete oldest file
    if oldest is not None and len(os.listdir(directory)) > threshold:
        os.remove(oldest[0])
        print("We have delete oldest file：", oldest[0])


def simple_lookup(y0, y1, treatment):
    assert y0.__len__() == y1.__len__()
    assert y0.__len__() == treatment.__len__()
    yf, ycf = [], []
    for ind, v in enumerate(treatment):
        if v == 1:
            yf.append(y1[ind])
            ycf.append(y0[ind])
        else:
            yf.append(y0[ind])
            ycf.append(y1[ind])
    return pd.Series(yf), pd.Series(ycf)


def cim_logger(log_file):
    logger = logging.getLogger('CIM')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    file_handler = logging.FileHandler(filename=log_file)
    file_handler.setLevel(level=logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)