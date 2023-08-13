import torch
import torch.nn as nn
import torch.nn.functional as F


# nn：Nerual Network的缩写，其中的Module类是所有模型的基类
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, x):

        # hidden conv layer
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # hidden conv layer
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # hidden linear layer
        x = x.reshape(-1, 12*4*4)
        x = self.fc1(x)
        x = F.relu(x)

        # hidden linear layer
        x = self.fc2(x)
        x = F.relu(x)

        # output layer
        y = self.out(x)
        # 如果损失函数使用的cross_entropy，则自带一个softmax函数，不需要自己添加
        # y = F.softmax(y, dim=1)

        return y


# network = Network()
# 类重写了print方法
# print(network)

# 利用 .+属性名来访问特定的层 后面还可以接.weight来访问参数
# print(network.conv1.weight)

# 卷积的参数是一个秩为4的张量，其中第一个维度的含义是卷积核的个数
#                            第二个维度的含义是每个卷积核的深度
#                             最后两个维度是单片卷积核的高和宽


