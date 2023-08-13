import torch

from build_model import Network
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

train_load = DataLoader(train_set, batch_size=10)

batch = next(iter(train_load))

network = Network()

images, labels = batch

with torch.no_grad():
    preds = network(images)
    print(preds.shape)
