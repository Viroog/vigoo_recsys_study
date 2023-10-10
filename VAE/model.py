import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, user_nums, item_nums, hidden_size, drop_ration):
        super(VAE, self).__init__()
        self.user_nums = user_nums
        self.item_nums = item_nums
        self.hidden_size = hidden_size
        self.drop_ration = drop_ration

        self.dropout = nn.Dropout(p=self.drop_ration)

        self.fc1 = nn.Linear(self.item_nums, 1000)
        self.fc2 = nn.Linear(1000, 500)

        self.mu_infer = nn.Linear(500, self.hidden_size)
        self.logvar_infer = nn.Linear(500, self.hidden_size)

        self.fc3 = nn.Linear(self.hidden_size, 500)
        self.fc4 = nn.Linear(500, 1000)

        self.decoder = nn.Linear(1000, self.item_nums)

        self.cuda()

    def forward(self, input_mat):
        if self.training:
            input_mat = self.dropout(input_mat)

        # fc1 & fc2
        input_mat = torch.tanh(self.fc1(input_mat))
        input_mat = torch.relu(self.fc2(input_mat))

        mu = self.mu_infer(input_mat)
        # 论文中指出在实现中，这里是output the log of the variance of the variational distribution,即log(var)
        logvar = self.logvar_infer(input_mat)

        # 训练的时候要加入方差，也就是噪音
        if self.training:
            # std=var^0.5=e^(0.5*log(var))
            std = torch.exp(0.5 * logvar)
            # 从标准正太分布中随机生成eps，形状和std一样
            eps = torch.randn_like(std)
            # 编码z (batch_size, hidden_size)
            z = mu + eps * std
        # 测试的时候z=mu即可，不用加入噪音
        else:
            z = mu

        # fc3 & fc4
        z = F.relu(self.fc3(z))
        z = F.tanh(self.fc4(z))

        # 恢复
        out = self.decoder(z)
        # 对于每一个用户的输入，即一个multi hot vector做log_softmax，得到概率
        prob = torch.log_softmax(out, dim=1)

        return prob, mu, logvar

    def predict(self, loader):
        preds = np.zeros_like(loader.dataset.data)

        for batch in loader:
            user_ids, input_mat = batch

            input_mat = input_mat.float().cuda()

            # 只需要prob，即用户喜欢的概率
            prob, _, _ = self.forward(input_mat)

            # 将用户购买过的设置为最小值
            prob = prob.masked_fill(input_mat.bool(), float("-inf"))

            user_ids = user_ids.cpu().numpy()
            preds[user_ids] = prob.detach().cpu().numpy()

        return preds
