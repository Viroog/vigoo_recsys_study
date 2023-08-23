import numpy as np


# 只保留分数为4和5的(user, item)
class Data:
    def __init__(self, path='../data/ml-100k/'):
        self.path = path

    def get_data(self):

        train_file = self.path + 'u1.base'
        train_data = []

        with open(train_file) as f:
            for line in f.readlines():
                splited = line.split(' ')[0].split('\t')
                u_i_r = [int(i.strip()) for i in splited[:3]]

                # rating要大于等于4
                if u_i_r[-1] >= 4:
                    train_data.append(u_i_r)

        test_file = self.path + 'u1.test'
        test_data = []

        with open(test_file) as f:
            for line in f.readlines():
                splited = line.split(' ')[0].split('\t')
                u_i_r = [int(i.strip()) for i in splited[:3]]

                if u_i_r[-1] >= 4:
                    test_data.append(u_i_r)

        train_data, test_data = np.array(train_data), np.array(test_data)

        train_data, test_data = self.mapping(train_data, test_data)

        return train_data, test_data

    def mapping(self, train_data, test_data):
        user_dict, item_dict = {}, {}

        train_user, train_item = train_data[:, 0], train_data[:, 1]
        test_user, test_item = test_data[:, 0], test_data[:, 1]

        total_user = np.concatenate((train_user, test_user))
        total_item = np.concatenate((train_item, test_item))

        user_ids, item_ids = list(set(total_user)), list(set(total_item))

        for i, user_id in enumerate(user_ids):
            if user_id not in user_dict.keys():
                user_dict[user_id] = i

        for i, item_id in enumerate(item_ids):
            if item_id not in item_dict.keys():
                item_dict[item_id] = i

        train_result, test_result = [], []

        for i in range(len(train_data)):
            user, item, rating = train_data[i, :]
            train_result.append([user_dict[user], item_dict[item], rating])

        for i in range(len(test_data)):
            user, item, rating = test_data[i, :]
            test_result.append([user_dict[user], item_dict[item], rating])

        return np.array(train_result), np.array(test_result)

