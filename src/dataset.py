from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import config
import torch


class SentimentClassifierDataset(Dataset):

    def __init__(self, train=True, text=None):
        self.train = train
        if self.train == True:
            self.data = pd.read_csv(config.TRAIN_DATA_PATH).iloc[:, 0].values
            self.labels = pd.read_csv(config.TRAIN_DATA_PATH).iloc[:, 1].values
        else:
            self.data = text
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
        self.n_samples = len(self.data)

    def _truncate_tokens(self, data):
        data = list(data)
        l = self.max_len // 2
        return data[:l] + data[-l:]

    def get_tokens(self, text):
        return self.tokenizer.encode(text, add_special_tokens=True)

    def _padding(self, data):
        data = list(data)
        return np.array(data + [0] * (self.max_len - len(data)))

    def get_attention_mask(self, data):
        return np.where(data != 0, 1, 0)

    def __getitem__(self, index):
        tokens = self.get_tokens(self.data[index])
        if len(tokens) > self.max_len:
            tokens = self._truncate_tokens(tokens)
        input_ids = self._padding(tokens)
        attention_mask = self.get_attention_mask(input_ids)
        if self.train == True:
            label = self.labels[index]
            res = {'input_ids': input_ids,
                   'attention_mask': attention_mask, 'label': label}
        else:
            res = {'input_ids': input_ids, 'attention_mask': attention_mask}
        return res

    def __len__(self):
        return self.n_samples
