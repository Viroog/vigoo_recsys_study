import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

# 准备数据一般遵循ETL过程，即extract、transform和load
# extract：从源中获取数据
# transform：将数据转变为tensor的形式
# load：将数据放入一个使其易于访问的对象中

# extract & transform
train_set = torchvision.datasets.FashionMNIST(
    root='./Data/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

# load
train_loader = DataLoader(train_set, batch_size=10, shuffle=True)



