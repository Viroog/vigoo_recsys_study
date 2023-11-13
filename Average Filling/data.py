import numpy as np


class Data:
    # 根据课件制定默认的路径，可以通过参数改变
    def __init__(self, path='../Data/ml-100k', file_name='u1'):
        self.train_file_path = path + '/' + file_name + '.base'
        self.test_file_path = path + '/' + file_name + '.test'

        self.get_data()

    def get_data(self):
        train_data, test_data = [], []

        with open(self.train_file_path) as f:
            for line in f.readlines():
                splited = line.split(' ')
                train_data.append([int(i) for i in splited[0].split('\t')[:3]])
        f.close()

        with open(self.test_file_path) as f:
            for line in f.readlines():
                splited = line.split(' ')
                test_data.append([int(i) for i in splited[0].split('\t')[:3]])
        f.close()

        train_data, test_data = np.array(train_data), np.array(test_data)

        return train_data, test_data
