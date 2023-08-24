import numpy as np


class PopRank:
    def __init__(self, data):

        self.user_nums, self.item_nums = len(set(data[:, 0])), len(set(data[:, 1]))

        self.mu = None
        self.data = data
        self.b_i = {}

    def train(self, return_val=1):
        self.i_u_dict = {}
        self.u_i_dict = {}

        for i in range(len(self.data)):
            user, item, rating = self.data[i, :]

            if user not in self.u_i_dict.keys():
                self.u_i_dict[user] = [item]
            else:
                self.u_i_dict[user].append(item)

            if item not in self.i_u_dict.keys():
                self.i_u_dict[item] = [user]
            else:
                self.i_u_dict[item].append(user)

        for item in self.data[:, 1]:
            self.b_i[item] = len(self.i_u_dict[item]) / self.user_nums

        self.b_i = dict(sorted(self.b_i.items(), key=lambda x: x[1], reverse=True))

        if return_val:
            return self.u_i_dict, self.b_i

    def test(self, data):

        pred_rec, real_buy = {}, {}

        for i in range(len(data)):
            user, item, rating = data[i, :]

            if user not in real_buy.keys():
                real_buy[user] = [item]
            else:
                real_buy[user].append(item)

            # 只需要预测一次
            if user not in pred_rec.keys():
                result = []

                # 对未评价过的物品进行评分预测
                for (k, v) in self.b_i.items():

                    if k not in self.u_i_dict[user]:
                        result.append(k)

                pred_rec[user] = result

        return pred_rec, real_buy
