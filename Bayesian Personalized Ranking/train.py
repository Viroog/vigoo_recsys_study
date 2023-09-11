from data import Data
from model import BPR

dataloader = Data("../data/ml-100k/")
print("finish load data!")
gamma, alpha_u, alpha_v, beta_v, T, d = 0.01, 0.01, 0.01, 0.01, 500, 20
bpr = BPR(dataloader, gamma, alpha_u, alpha_v, beta_v, T, d)
bpr.train()
bpr.test()
