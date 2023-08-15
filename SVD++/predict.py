from model import SVDpp
from data import LoadData

# 加载数据
train_data, test_data = LoadData(path='../data/ml-1m/ratings.dat').get_data()
svd = SVDpp(train_data)
svd.train()
svd.test(data=test_data)