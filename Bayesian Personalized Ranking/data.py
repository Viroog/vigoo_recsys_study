from collections import defaultdict

import numpy as np


class Data:
    def __init__(self, path):
        self.path = path
        self.train_data, self.test_data = self.get_data()
        self.train_u_i_dict, self.test_u_i_dict, self.train_i_u_dict = self.get_u_i_dict()
        print(self.max_u, self.max_i)

    def get_data(self):

        train_path = self.path + "u1.base"
        train_data = []

        with open(train_path, 'r') as f:
            for line in f.readlines():
                # 这里的rating筛选那些评分>=4的，在训练时用不到
                u, i, rating, _ = line.split('\t')
                # train_data.append([int(u), int(i)])
                if rating == '4' or rating == '5':
                    train_data.append([int(u), int(i)])

        test_path = self.path + "u1.test"
        test_data = []

        with open(test_path, 'r') as f:
            for line in f.readlines():
                u, i, rating, _ = line.split('\t')
                # test_data.append([int(u), int(i)])
                if rating == '4' or rating == '5':
                    test_data.append([int(u), int(i)])

        train_data, test_data = np.array(train_data), np.array(test_data)

        return train_data, test_data

    def get_u_i_dict(self):
        train_data, test_data = self.train_data, self.test_data

        max_u, max_i = -1, -1

        # defaultdict的两个好处：
        # 1.可以制定任意的类型
        # 2.在查询不在的键时，会返回空的list
        train_u_i_dict, test_u_i_dict, train_i_u_dict = defaultdict(list), defaultdict(list), defaultdict(list)

        for i in range(len(train_data)):
            u, i = train_data[i, 0], train_data[i, 1]

            max_u = max(max_u, u)
            max_i = max(max_i, i)

            train_u_i_dict[u].append(i)

            train_i_u_dict[i].append(u)

        for i in range(len(test_data)):
            u, i = test_data[i, 0], test_data[i, 1]

            max_u = max(max_u, u)
            max_i = max(max_i, i)

            test_u_i_dict[u].append(i)

        self.max_u, self.max_i = max_u, max_i

        return train_u_i_dict, test_u_i_dict, train_i_u_dict
