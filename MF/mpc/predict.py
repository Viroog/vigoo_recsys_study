from data import LoadData
from model import MF_MPC

train_data, test_data = LoadData(path="../../data/ml-1m/ratings.dat").get_data()
mf_mpc = MF_MPC(data=train_data, d=20)
mf_mpc.train()
mf_mpc.test(data=test_data)

