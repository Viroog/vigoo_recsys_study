import torch
import torch.nn as nn

import numpy as np


# 这里实现两种模型: 1) 按照论文的模型实现
#                2) 利用Pytorch中的TransformerEncoderLayer和TransformerEncoder构建
# 两个模型的主要差距在于Dropout和Layer Norm的时机，前馈神经网络的架构也不太一样
class SASRec(nn.Module):
    def __init__(self, n, item_nums, d, dropout_rate, block_nums, head_nums):
        super(SASRec, self).__init__()
        # n为能够处理的序列最大长度
        self.n = n
        self.item_nums = item_nums
        # d为latent dimension
        self.d = d
        self.dropout_rate = dropout_rate
        # self-attention+ffn为一个block
        self.block_nums = block_nums
        # 自注意力头数
        self.head_nums = head_nums

        # 下三角形的单位矩阵，并将数据放到gpu上
        self.mask = torch.tril(torch.ones((self.n, self.n))).cuda()

        self.item_embedding = nn.Embedding(self.item_nums + 1, self.d)
        self.position_embedding = nn.Embedding(self.n, self.d)
        self.dropout = nn.Dropout(p=self.dropout_rate)

        # 里面带有一个self-attention和一个FFN
        # 将batch_fist设置为True，才是(batch_size, seq, feature_dim)，要不然是(seq, batch_size, feature_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d, nhead=self.head_nums, dim_feedforward=self.n, dropout=self.dropout_rate, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=self.block_nums)

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
        position_embed = self.position_embedding(position)

        # shape: (batch_size, n, d)
        input_embed = self.dropout(seq_embed + position_embed)

        # 要有一个mask，使得attention中过去的能影响未来，而不能未来影响过去
        # 这里的输出对应模型最上方的向量
        # shape: (batch_size, n, d)
        output_embed = self.encoder(input_embed, self.mask)

        # 正样本和负样本的embedding
        pos_embed = self.item_embedding(pos)
        neg_embed = self.item_embedding(neg)

        # 预测公式：r_(i,t) = F_t * M_i
        # 并对最后一维求和，即获得预测分数
        # shape: (batch_size, n)
        pos_pred = (pos_embed * output_embed).sum(axis=-1)
        neg_pred = (neg_embed * output_embed).sum(axis=-1)

        return pos_pred, neg_pred

    def predict(self, seq):
        pass

