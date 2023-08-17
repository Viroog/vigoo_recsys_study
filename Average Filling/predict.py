from data import Data
from model import AF

train_data, test_data = Data().get_data()
af = AF(data=train_data)
af.train()
af.test(data=test_data)
