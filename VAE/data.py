import numpy as np
import pandas as pd
import torch.utils.data as data
from sklearn.model_selection import train_test_split


class Data:
    # 数据集是ml-1m，path直接具体到文件名
    def __init__(self, path, test_size=0.3):
        self.path = path
        self.test_size = test_size
        self.train_data, self.test_data = None, None
        self.user_nums, self.item_nums = None, None
        self.get_data()

    # explicit feedback -> implicit feedback
    def get_data(self):

        data = []

        if self.path == '../Data/ml-1m/ratings.dat':
            with open(self.path, 'r') as f:
                for line in f.readlines():
                    splited = line.split("::")
                    user, item, rating, _ = splited
                    # user和item序号减1，从0开始
                    data.append([int(user) - 1, int(item) - 1, int(rating)])
        elif self.path == '../Data/ml-20m/ratings.csv':
            df = pd.read_csv(self.path, encoding='gbk')

            for i in range(len(df)):
                user, item, rating, _ = df.iloc[i, :]
                data.append([int(user) - 1, int(item) - 1, rating])

        data = np.array(data)

        self.user_nums, self.item_nums = np.max(data[:, 0]) + 1, np.max(data[:, 1]) + 1
        train_data, test_data = train_test_split(data, test_size=self.test_size, random_state=42, shuffle=True)

        # 除去出现在测试集中，但未出现过在训练集中的物品
        # 需要移除的物品
        train_items, test_items = list(set(train_data[:, 1])), list(set(test_data[:, 1]))

        removed_items = []
        for item in test_items:
            if item not in train_items:
                removed_items.append(item)

        updated_test_data = []
        for i in range(len(test_data)):
            user, item, rating = test_data[i, :]
            if item not in removed_items:
                updated_test_data.append([user, item, rating])

        updated_test_data = np.array(updated_test_data)

        # (6040, 3952)
        self.train_data = self.to_implicit_matrix(train_data)
        self.test_data = self.to_implicit_matrix(updated_test_data)



    def to_implicit_matrix(self, data):
        implicit_matrix = np.zeros((self.user_nums, self.item_nums))

        users, items = data[:, 0], data[:, 1]
        implicit_matrix[users, items] = 1.0

        return implicit_matrix


class DataSet(data.Dataset):
    def __init__(self, data):
        super(DataSet, self).__init__()

        self.data = data

    # 重写方法
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return index, self.data[index]
