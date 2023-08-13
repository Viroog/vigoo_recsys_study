import torch

from build_model import Network
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader as Dataloader

train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

# train_load = Dataloader()


network = Network()


sample = next(iter(train_set))
image, label = sample

image = image.unsqueeze(dim=0)

with torch.no_grad():
    pred = network(image)
    print(pred)
