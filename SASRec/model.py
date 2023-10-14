import torch.nn as nn


class SASRec(nn.Module):
    def __init__(self, n, item_nums, d, dropout_rate):
        # n为能够处理的序列最大长度
        self.n = n
        self.item_nums = item_nums
        # d为latent dimension
        self.d = d
        self.dropout_rate

        self.item_embedding = nn.Embedding(self.item_nums, self.d)
        self.position_embedding = nn.Embedding(self.n, self.d)

    def forward(self, input_mat):
        pass

    def test(self):
        pass
