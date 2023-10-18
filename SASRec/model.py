import numpy as np
import torch
import torch.nn as nn


#
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

        # self.mask = (torch.triu(torch.ones((self.n, self.n))) == 0).cuda()

        self.item_embedding = nn.Embedding(self.item_nums + 1, self.d, padding_idx=0)
        self.position_embedding = nn.Embedding(self.n, self.d)
        self.dropout = nn.Dropout(p=self.dropout_rate)

        # 里面带有一个self-attention和一个FFN
        # 将batch_fist设置为True，才是(batch_size, seq, feature_dim)，要不然是(seq, batch_size, feature_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d, nhead=self.head_nums, dim_feedforward=self.d,
                                                        dropout=self.dropout_rate, batch_first=True)
        # 这里效果差了一点的原因可能是少经过了一层FFN
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=self.block_nums)

    def forward(self, seq, pos, neg):
        # shape: (batch_size, n)
        seq, pos, neg = torch.LongTensor(seq).cuda(), torch.LongTensor(pos).cuda(), torch.LongTensor(neg).cuda()

        # shape: (batch_size, n, d)
        seq_embed = self.item_embedding(seq)
        # np.tile用于在某个维度上复试数组内容
        # reps代表每个维度上复制次数的参数 [dim0, dim1, ...]
        # shape: (batch_size, n)
        position = np.tile(np.array(range(seq.shape[1])), reps=[seq.shape[0], 1])
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

    def predict(self, seq, item_idxs):
        seq, item_idxs = torch.LongTensor(seq).cuda(), torch.LongTensor(item_idxs).cuda()

        # shape: (1, n, d)
        seq_embed = self.item_embedding(seq)

        # shape: (1, n)
        position = np.tile(np.array(range(seq.shape[1])), reps=[seq.shape[0], 1])
        position = torch.LongTensor(position).cuda()
        # shape: (1, n, d)
        position_embed = self.position_embedding(position)

        input_embed = self.dropout(seq_embed + position_embed)

        # shape: (1, n, d)
        output = self.encoder(input_embed, self.mask)

        # self-attention的最后一个输出，即用户偏好的集大成者
        # shape: (1, 50)
        last_output = output[:, -1, :]

        # shape: (101, 50)
        item_embeds = self.item_embedding(item_idxs)

        # 第一个unsqueeze是让其从一维向量变成二维矩阵进行相乘
        # 矩阵相乘后的shape: (1, 101, 1)
        # 再进行维度压缩的shape: (1, 101)
        preds = item_embeds.matmul(last_output.unsqueeze(-1)).squeeze(-1)

        return preds


# 源码使用的是一维卷积代替全连接网络
class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_nums, dropout_rate, use_conv=True):
        super(PointWiseFeedForward, self).__init__()

        self.hidden_nums = hidden_nums
        self.dropout_rate = dropout_rate
        self.use_conv = use_conv

        # used_conv用来选择是否用卷积层来代替全连接层
        if use_conv:
            # 卷积核的大小由kernel_size*in_channel来决定
            # 输入维度和输出维度不变
            self.fc1 = nn.Conv1d(in_channels=self.hidden_nums, out_channels=self.hidden_nums, kernel_size=1)
            self.fc2 = nn.Conv1d(in_channels=self.hidden_nums, out_channels=self.hidden_nums, kernel_size=1)
        else:
            self.fc1 = nn.Linear(self.hidden_nums, self.hidden_nums)
            self.fc2 = nn.Linear(self.hidden_nums, self.hidden_nums)

        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.dropout2 = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        # 一维卷积的输入应该是(N, Cin, L), 其中N为batch_size, Cin为通道数，L为序列长度，因此要先将输入的最后两个维度交换
        if self.use_conv:
            x = x.transpose(-1, -2)
        # 每经过一个fc就要先dropout，再relu
        output = self.dropout2(self.fc2(self.relu(self.dropout1(self.fc1(x)))))
        if self.use_conv:
            # 将形状变回来
            output = output.transpose(-1, -2)
            # x也要转回来
            x = x.transpose(-1, -2)
        # residual
        output += x

        return output


class SASRec2(nn.Module):
    def __init__(self, n, item_nums, d, dropout_rate, block_nums, head_nums, use_conv=True):
        super(SASRec2, self).__init__()

        self.n = n
        self.item_nums = item_nums
        self.d = d
        self.dropout_rate = dropout_rate
        self.block_nums = block_nums
        self.head_nums = head_nums
        self.use_conv = use_conv

        self.item_embed = nn.Embedding(self.item_nums + 1, self.d, padding_idx=0)
        self.position_embed = nn.Embedding(self.n, self.d)
        self.embed_dropout = nn.Dropout(self.dropout_rate)

        # self-attention之前的layer norm
        self.Attention_normed = nn.ModuleList()
        # dropout在MutiheadAttention中自带
        self.Attention = nn.ModuleList()
        # FFN之前的layer norm
        self.FFN_normed = nn.ModuleList()
        self.FFN = nn.ModuleList()

        for _ in range(self.block_nums):
            Attention_normed = nn.LayerNorm(self.d, eps=1e-8)
            self.Attention_normed.append(Attention_normed)

            # 这里我加了batch_fist，源代码中没有加，因此需要调换顺序
            Attention = nn.MultiheadAttention(embed_dim=self.d, num_heads=self.head_nums, dropout=self.dropout_rate,
                                              batch_first=True)
            self.Attention.append(Attention)

            FFN_normed = nn.LayerNorm(self.d, eps=1e-8)
            self.FFN_normed.append(FFN_normed)

            FFN = PointWiseFeedForward(self.d, self.dropout_rate, use_conv=self.use_conv)
            self.FFN.append(FFN)

        self.last_layer_normed = nn.LayerNorm(self.d, eps=1e-8)

    def log2feat(self, seq):
        # shape: (batch_size, n, d)
        seq_embed = self.item_embed(torch.LongTensor(seq).cuda())
        # 标准化
        seq_embed *= self.d ** 0.5
        # shape: (batch_size, n)
        position = np.tile(np.array(range(seq.shape[1])), reps=[seq.shape[0], 1])
        position = torch.LongTensor(position).cuda()
        # shape: (batch_size, n, d)
        position_embed = self.position_embed(position)
        input_embed = self.embed_dropout(seq_embed + position_embed)

        # 因为加上了position_embed，使得padding的embed不为0，让其恢复为0
        timeline_mask = torch.BoolTensor(seq == 0).cuda()
        # 原本timeline_mask的shape为(batch_size, n)，现在加多一维度，变成(batch_size, n, 1)，在最后一维上进行广播
        input_embed *= ~timeline_mask.unsqueeze(-1)

        for i in range(self.block_nums):

            normed_input_embed = self.Attention_normed[i](input_embed)

            # 这个mask是对计算出的d*d矩阵进行操作的，所以形状也应该为d*d
            attention_mask = torch.tril(torch.ones(self.n, self.n)).cuda()

            attention_output, _ = self.Attention[i](normed_input_embed, input_embed, input_embed, attn_mask=attention_mask)

            # residual
            seqs = attention_output + normed_input_embed

            seqs = self.FFN_normed[i](seqs)
            seqs = self.FFN[i](seqs)
            # 再一次置0
            seqs *= ~timeline_mask.unsqueeze(-1)

        output = self.last_layer_normed(seqs)

        return output

    def forward(self, seq, pos, neg):
        # input shape: (batch_size, n)

        # shape: (batch_size, n, d)
        output = self.log2feat(seq)

        pos_embed = self.item_embed(torch.LongTensor(pos).cuda())
        neg_embed = self.item_embed(torch.LongTensor(neg).cuda())

        pos_pred = (output * pos_embed).sum(axis=-1)
        neg_pred = (output * neg_embed).sum(axis=-1)

        return pos_pred, neg_pred

    def predict(self, seq, item_idxs):
        # shape: (1, n, d)
        output = self.log2feat(seq)
        # 取出最后一个向量(1, d)
        last_output = output[:, -1, :]

        # shape: (101, d)
        item_embed = self.item_embed(torch.LongTensor(item_idxs).cuda())

        # shape: (1, 101)
        pred = last_output.matmul(item_embed.T)

        return pred
