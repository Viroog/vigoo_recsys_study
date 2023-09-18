from FISM.data import Data
from model import FISMrmse

dataloader = Data(path='../../data/ml-100k/')
model = FISMrmse(dataloader, 0.5, 0.01, 3, 20, 0.001, 0.001, 0.001, 0.001, 100)
model.train()
model.test()
