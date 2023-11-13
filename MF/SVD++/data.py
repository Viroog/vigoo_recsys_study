import numpy as np
from random import shuffle


# 该类用于加载数据
class LoadData:
    def __init__(self, path):
        self.path = path

    # 从文件中获取数据
    def get_data(self):

        data = []

        with open(self.path) as f:
            for line in f.readlines():
                # user_id::item_id::rating
                splited = line.split("::")
                data.append([int(i) for i in splited[:3]])

        shuffle(data)

        # 划分训练集和测试集, 7:3
        train_data, test_data = data[:int(len(data) * 0.7)], data[int(len(data)*0.7):]

        # pre-processing
        train_data, test_data = self.mapping(train_data), self.mapping(train_data)
        print('load Data finish!')
        return train_data, test_data

    # 原数据的id不是从0开始，将id调整为从0开始的，连续的id
    # 理由：要构建implicit feedback的矩阵(binary matrix)，降低矩阵的维度，否则矩阵太大
    def mapping(self, data):
        # 要将第一列和第二列单独取出来重新映射(利用set去重以及字典映射)，转为ndarray更方便操作
        data = np.array(data)
        user_ids, item_ids = list(set(data[:, 0])), list(set(data[:, 1]))

        user_dict = {}
        for i, user_id in enumerate(user_ids):
            user_dict[user_id] = i

        item_dict = {}
        for i, item_id in enumerate(item_ids):
            item_dict[item_id] = i

        # 映射过后的数据
        mapping_data = []
        for i in range(len(data)):
            user_id, item_id, rating = data[i]
            mapping_data.append([user_dict[user_id], item_dict[item_id],rating])

        return mapping_data


# for test
# train_data, test_data = LoadData(path='../Data/ml-1m/ratings.dat').get_data()
