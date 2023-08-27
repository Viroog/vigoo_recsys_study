import numpy as np


class Data:
    def __init__(self, path):
        self.path = path

    def get_data(self):

        train_path = self.path + "u1.base"
        train_data = []

        # 只保留分数4和5的
        with open(train_path) as f:
            for line in f.readlines():
                splited = line.split(' ')[0].split('\t')[:3]
                if splited[-1] == '4' or splited[-1] == '5':
                    train_data.append([int(i) for i in splited])

        test_path = self.path + "u1.test"
        test_data = []

        with open(test_path) as f:
            for line in f.readlines():
                splited = line.split(' ')[0].split('\t')[:3]
                if splited[-1] == '4' or splited[-1] == '5':
                    test_data.append([int(i) for i in splited])

        train_data, test_data = self.mapping(np.array(train_data), np.array(test_data))

        return train_data, test_data

    def mapping(self, train_data, test_data):

        train_user, train_item = list(set(train_data[:, 0])), list(set(train_data[:, 1]))
        test_user, test_item = list(set(test_data[:, 0])), list(set(test_data[:, 1]))

        train_user.extend(test_user)
        train_item.extend(test_item)

        total_user, total_item = train_user, train_item

        user_dict, item_dict = {}, {}

        for user in total_user:
            if user not in user_dict.keys():
                user_dict[user] = len(user_dict)

        for item in total_item:
            if item not in item_dict.keys():
                item_dict[item] = len(item_dict)

        for i in range(len(train_data)):
            train_data[i, 0], train_data[i, 1] = user_dict[train_data[i, 0]], item_dict[train_data[i, 1]]

        for i in range(len(test_data)):
            test_data[i, 0], test_data[i, 1] = user_dict[test_data[i, 0]], item_dict[test_data[i, 1]]

        return train_data, test_data
