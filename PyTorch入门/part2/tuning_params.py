# 打开前端指令: tensorboard --logdir=runs

import torch
from build_model import Network
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from itertools import product

parameters = dict(
    lr=[0.01, 0.001],
    batch_size=[10,100,1000],
    shuffle=[True, False]
)

param_values = [v for v in parameters.values()]

# product函数会产生所有可能的组合
# *param_values表示将里面的每一个元素看成一个参数，而不是将整个param_values看成一个参数
for lr, batch_size, shuffle in product(*param_values):

    train_set = torchvision.datasets.FashionMNIST(
        root='./data/FashionMNIST',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)

    # 注释字符串，用于在tensorboard中唯一标识
    comment = f'batch_size={batch_size} lr={lr}'
    # 创建一个SummaryWriter实例
    tb = SummaryWriter(comment=comment)

    network = Network()
    images, labels = next(iter(train_loader))
    grid = torchvision.utils.make_grid(images)

    tb.add_image('images', grid)
    tb.add_graph(network, images)
    # 像文件一样，写完就要关闭
    tb.close()

    optimizer = optim.Adam(network.parameters(), lr=lr)

    for epoch in range(10):

        total_loss, total_correct = 0, 0

        for batch in train_loader:
            images, labels = batch

            preds = network(images)
            loss = F.cross_entropy(preds, labels)

            # 吧参数权重记得要清0
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # 在不同的batch规模下，使得total_loss具有可比性
            total_loss += loss.item() * batch_size
            total_correct += (preds.argmax(dim=1) == labels).sum()

        # 标量 三个参数：key value epcoh
        tb.add_scalar('Loss', total_loss, epoch)
        tb.add_scalar('Number Correct', total_correct, epoch)
        tb.add_scalar('Accuracy', total_correct / len(train_set), epoch)

        # 直方图同理
        # 不是通用方法
        # tb.add_histogram('conv1.bias', network.conv1.bias, epoch)
        # tb.add_histogram('conv1.weight', network.conv1.weight, epoch)
        # tb.add_histogram('conv1.weight.grad', network.conv1.weight.grad, epoch)

        for name, weight in network.named_parameters():
            tb.add_histogram(name, weight, epoch)
            tb.add_histogram(f'{name}.grad', weight.grad, epoch)

        print(f"epoch: {epoch + 1}, total_loss: {total_loss}, total_correct: {total_correct}")

    tb.close()
