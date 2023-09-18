import numpy as np
from collections import defaultdict


class Data:
    def __init__(self, path):
        self.path = path
        self.train_data, self.test_data = self.get_data()
        self.train_u_i_dict, self.test_u_i_dict = self.get_u_i_dict()

    def get_data(self):

        train_path = self.path + 'u1.base'
        train_data = []

        with open(train_path, 'r') as f:
            for line in f.readlines():
                u, i, rating, _ = line.split('\t')

                if rating == '4' or rating == '5':
                    train_data.append([int(u), int(i)])

        test_path = self.path + 'u1.test'
        test_data = []

        with open(test_path, 'r') as f:
            for line in f.readlines():
                u, i, rating, _ = line.split('\t')

                if rating == '4' or rating == '5':
                    test_data.append([int(u), int(i)])

        train_data, test_data = np.array(train_data), np.array(test_data)

        return train_data, test_data

    def get_u_i_dict(self):

        train_u_i_dict, test_u_i_dict = defaultdict(list), defaultdict(list)

        train_data, test_data = self.train_data, self.test_data

        max_u, max_i = -1, -1

        for i in range(len(train_data)):
            u, i = train_data[i, :]

            max_u = max(max_u, u)
            max_i = max(max_i, i)

            train_u_i_dict[u].append(i)

        for i in range(len(test_data)):
            u, i = test_data[i, :]

            max_u = max(max_u, u)
            max_i = max(max_i, i)

            test_u_i_dict[u].append(i)

        # # 把值的长度只有1的键值对去除掉
        # train_u_i_dict = {k: v for k, v in train_u_i_dict.items() if len(v) > 1}
        # test_u_i_dict = {k: v for k, v in test_u_i_dict.items() if len(v) > 1}

        self.max_u = max_u
        self.max_i = max_i

        return train_u_i_dict, test_u_i_dict
