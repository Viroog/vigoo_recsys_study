# 打开前端指令: tensorboard --logdir=runs

import torch

from build_model import Network
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

train_loader = DataLoader(train_set, batch_size=100)


# 创建一个SummaryWriter实例
tb = SummaryWriter()

network = Network()
images, labels = next(iter(train_loader))
grid = torchvision.utils.make_grid(images)

tb.add_image('images', grid)
tb.add_graph(network, images)
# 像文件一样，写完就要关闭
tb.close()


optimizer = optim.Adam(network.parameters(), lr=0.01)

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

        total_loss += loss.item()
        total_correct += (preds.argmax(dim=1) == labels).sum()

    # 标量 三个参数：key value epcoh
    tb.add_scalar('Loss', total_loss, epoch)
    tb.add_scalar('Number Correct', total_correct, epoch)
    tb.add_scalar('Accuracy', total_correct / len(train_set), epoch)

    # 直方图同理
    tb.add_histogram('conv1.bias', network.conv1.bias, epoch)
    tb.add_histogram('conv1.weight', network.conv1.weight, epoch)
    tb.add_histogram('conv1.weight.grad', network.conv1.weight.grad, epoch)


    print(f"epoch: {epoch+1}, total_loss: {total_loss}, total_correct: {total_correct}")

tb.close()