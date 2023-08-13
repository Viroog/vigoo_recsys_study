import torch

from build_model import Network
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)
train_load = DataLoader(train_set, batch_size=100)

network = Network()

optimizer = optim.Adam(network.parameters(), lr=0.01)

max_epoch = 1000

for epoch in range(max_epoch):

    total_loss, total_correct = 0, 0

    for batch in train_load:
        images, labels = batch

        preds = network(images)
        loss = F.cross_entropy(preds, labels)

        # 吧参数权重记得要清0
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        total_correct += (preds.argmax(dim=1) == labels).sum()

    print(f"epoch: {epoch+1}, total_loss: {total_loss}, total_correct: {total_correct}")