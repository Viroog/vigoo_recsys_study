import torch
import torch.nn as nn

import numpy as np


class SASRec(nn.Module):
    def __init__(self, n, item_nums, d, dropout_rate):
        super(SASRec, self).__init__()
        # n为能够处理的序列最大长度
        self.n = n
        self.item_nums = item_nums
        # d为latent dimension
        self.d = d
        self.dropout_rate = dropout_rate

        self.item_embedding = nn.Embedding(self.item_nums + 1, self.d)
        self.position_embedding = nn.Embedding(self.n, self.d)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.cuda()

    def forward(self, seq, pos, neg):
        # shape: (batch_size, n)
        seq, pos, neg = torch.LongTensor(seq).cuda(), torch.LongTensor(pos).cuda(), torch.LongTensor(neg).cuda()

        # shape: (batch_size, n, d)
        seq_embed = self.item_embedding(seq)
        # np.tile用于在某个维度上复试数组内容
        # reps代表每个维度上复制次数的参数 [dim0, dim1, ...]
        # shape: (batch_size, n)
        position = np.tile(np.array(range(pos.shape[1])), reps=[pos.shape[0], 1])
        position = torch.LongTensor(position).cuda()
        # shape: (batch_size, n, d)
        pos_embed = self.position_embedding(position)

        # shape: (batch_size, n, d)
        input = seq_embed + pos_embed

        # 接下来经过self-attention
        pass

    def test(self):
        pass
