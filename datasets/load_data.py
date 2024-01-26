# coding: utf-8
# @Time: 2024/1/26 13:42
# @Author: **
# @FileName: load_data.py
# @Software: PyCharm Community Edition
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from DDRN.utils.utils import simple_lookup


class LoadData(object):
    def __init__(self, data_src, ratio=0.75, index=1):
        self.data_src = data_src
        self.ratio = ratio
        self.index = index

    def read_acic(self
                  ,data_folder
                  ,sep):
        x_data = pd.read_csv(data_folder+'x.csv', sep=sep)
        zymu_data = pd.read_csv(data_folder+'zymu_{}.csv'.format(self.index))

        non_numeric_cols = x_data.select_dtypes(include=[object]).columns
        # Be mindful of the Python version
        X = pd.get_dummies(x_data, columns=non_numeric_cols, drop_first=True, dtype=int)
        t = zymu_data['z'].astype(np.float32)
        yf, ycf = simple_lookup(zymu_data['y0'], zymu_data['y1'], zymu_data['z'])
        mu0 = zymu_data['mu0'].astype(np.float32)
        mu1 = zymu_data['mu1'].astype(np.float32)
        data = [t, yf.astype(np.float32), ycf.astype(np.float32), mu0, mu1, X]

        return data

    def split_acic_data(self, data, batch_size):
        tts_data = train_test_split(data[0], data[1], data[2], data[3], data[4], data[5], train_size=self.ratio)
        # Train data iterator
        train_data = ACIC2016Dataset(t=tts_data[0], yf=tts_data[2], ycf=tts_data[4], mu0=tts_data[6], mu1=tts_data[8], x=tts_data[10])
        train_data_iterator = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        # Test data iterator
        test_data = ACIC2016Dataset(t=tts_data[1], yf=tts_data[3], ycf=tts_data[5], mu0=tts_data[7], mu1=tts_data[9], x=tts_data[11])
        test_data_iterator = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        return train_data_iterator, test_data_iterator


class ACIC2016Dataset(Dataset):
    def __init__(self, t, yf, ycf, mu0, mu1, x):
        self.t = torch.from_numpy(t.values.reshape(-1, 1))
        self.yf = torch.from_numpy(yf.values.reshape(-1, 1))
        self.ycf = torch.from_numpy(ycf.values.reshape(-1, 1))
        self.mu0 = torch.from_numpy(mu0.values.reshape(-1, 1))
        self.mu1 = torch.from_numpy(mu1.values.reshape(-1, 1))
        self.x = torch.from_numpy(x.values)

    def __len__(self):
        return self.x.size()[0]

    def __getitem__(self, index):
        return self.t[index, :], self.yf[index, :], self.ycf[index, :], self.mu0[index, :], self.mu1[index, :], self.x[index, :]
