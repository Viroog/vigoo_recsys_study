import numpy as np
import torch.utils.data as data
from sklearn.model_selection import train_test_split


class Data:
    # 数据集是ml-1m，path直接具体到文件名
    def __init__(self, path, test_size=0.2):
        self.path = path
        self.test_size = test_size
        self.train_data, self.test_data = None, None
        self.user_nums, self.item_nums = None, None
        self.get_data()

    # explicit feedback -> implicit feedback
    def get_data(self):

        data = []

        with open(self.path, 'r') as f:
            for line in f.readlines():
                splited = line.split("::")
                user, item, rating, _ = splited
                # user和item序号减1，从0开始
                data.append([int(user) - 1, int(item) - 1, int(rating)])

        data = np.array(data)

        self.user_nums, self.item_nums = np.max(data[:, 0]) + 1, np.max(data[:, 1]) + 1
        train_data, test_data = train_test_split(data, test_size=self.test_size, random_state=42, shuffle=True)

        # (6040, 3952)
        self.train_data = self.to_implicit_matrix(train_data)
        self.test_data = self.to_implicit_matrix(test_data)

    def to_implicit_matrix(self, data):
        implicit_matrix = np.zeros((self.user_nums, self.item_nums))

        for i in range(len(data)):
            user, item, rating = data[i, :]
            implicit_matrix[user, item] = 1

        return implicit_matrix


class DataSet(data.Dataset):
    def __init__(self, data):
        super(DataSet, self).__init__()

        self.data = data

    # 重现方法
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return index, self.data[index]
