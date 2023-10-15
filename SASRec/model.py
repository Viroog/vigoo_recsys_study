import torch.nn as nn


class SASRec(nn.Module):
    def __init__(self, n, item_nums, d, dropout_rate):
        super(SASRec, self).__init__()
        # n为能够处理的序列最大长度
        self.n = n
        self.item_nums = item_nums
        # d为latent dimension
        self.d = d
        self.dropout_rate = dropout_rate

        self.item_embedding = nn.Embedding(self.item_nums, self.d)
        self.position_embedding = nn.Embedding(self.n, self.d)

        self.cuda()

    def forward(self):
        pass

    def test(self):
        pass
