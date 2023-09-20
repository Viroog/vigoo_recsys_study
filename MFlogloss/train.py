from data import Data
from model import MFlogloss

dataloader = Data(path='../data/ml-100k/')
model = MFlogloss(dataloader, 3, 20, 0.01, 0.001, 0.001, 0.001, 0.001, 100)
model.train()
model.test()
