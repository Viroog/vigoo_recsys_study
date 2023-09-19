from FISM.data import Data
from model import FISMauc

dataloader = Data(path='../../data/ml-100k/')
model = FISMauc(dataloader, 0.5, 0.01, 3, 0.01, 0.01, 0.01, 20, 1000)
model.train()
model.test()
