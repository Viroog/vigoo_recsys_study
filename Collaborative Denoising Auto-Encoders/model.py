import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CDAE(nn.Module):
    # AutoEncoder的就是简单的三层全连接层：输入层，隐藏层和输出层
    def __init__(self, user_nums, item_nums, hidden_size, corrupted_ration):
        super(CDAE, self).__init__()
        self.user_nums = user_nums
        self.item_nums = item_nums
        self.hidden_size = hidden_size
        self.corrupted_ration = corrupted_ration

        # user_embedding, 维度是hidden_size，因为后面要和隐藏层的输出相加
        self.user_embedding = nn.Embedding(user_nums, hidden_size)
        self.dropout = nn.Dropout(p=self.corrupted_ration)
        # encoder-decoder就是先降维后升维，hidden_size远小于输入，用于寻找输入的模式
        # bias默认设置为True
        self.encoder = nn.Linear(item_nums, hidden_size)
        self.decoder = nn.Linear(hidden_size, item_nums)

        # 使用gpu训练
        self.cuda()

    # x是一个矩阵
    def forward(self, user_ids, input_mat):
        # 设置成cuda使用的变量形式
        user_ids = user_ids.cuda()
        input_mat = input_mat.float().cuda()

        # 先以1-p的概率对某些维度进行倍增
        dim_to_scale = torch.rand(input_mat.size(1)) > self.corrupted_ration

        scaled_input = input_mat.clone()
        scaled_input[:, dim_to_scale] *= 1 / (1 - self.corrupted_ration)

        # 再以p的概率添加noisy，即进行dropout
        dropout_scaled_input = self.dropout(scaled_input)

        hidden_output = torch.sigmoid(self.encoder(dropout_scaled_input) + self.user_embedding(user_ids))

        # dropout_input = F.dropout(input_mat, p=self.corrupted_ration, training=True)
        # hidden_output = F.sigmoid(self.encoder(dropout_input) + self.user_embedding(user_ids))

        output = self.decoder(hidden_output)

        return output

    def predict(self, loader):
        preds = np.zeros_like(loader.dataset.data)

        # 不建立计算图，节省内存
        with torch.no_grad():

            for batch in loader:
                user_ids, input_mat = batch

                user_ids = user_ids.cuda()
                input_mat = input_mat.float().cuda()

                hidden_output = torch.sigmoid(self.encoder(input_mat) + self.user_embedding(user_ids))
                output = torch.sigmoid(self.decoder(hidden_output))

                # 将在训练集中用户已经购买过的物品预测评分置为最小值
                output = output.masked_fill(
                    input_mat.bool(), float("-inf")
                )

                user_ids = user_ids.cpu().numpy()
                preds[user_ids] = output.cpu().numpy()

        return preds
