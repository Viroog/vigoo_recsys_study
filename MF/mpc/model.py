import numpy as np
import pandas as pd
from numba import jit


class MF_MPC:
    # 超参数默认为论文中ML1M的超参数
    def __init__(self, data, d=20):
        self.d = d

        # 加多了一个DataFrame是为了方便查询
        self.df_data = pd.DataFrame(data, columns=['user', 'item', 'rating'])
        self.data = np.array(data)

        # user_id和item_id是mapping过的，从0开始的连续整数
        self.user_nums, self.item_nums, self.rating_list = len(set(self.data[:, 0])), len(set(self.data[:, 1])), list(
            set(self.data[:, 2]))

        # 先使用随机数来初始化，看看效果如何
        # 后面与论文一致，只有统计学来初始化
        self.U = np.random.randn(self.user_nums, self.d)
        self.V = np.random.randn(self.item_nums, self.d)
        self.M = np.random.randn(self.item_nums, self.d)

        self.mu = self.data[:, 2].mean()
        self.b_u = np.random.randn(self.user_nums)
        self.b_i = np.random.randn(self.item_nums)

        # 用户给各个物品评分的记录字典, 结构为字典套字典
        # 'user': {
        #       'score 1': [],
        #       'score 2': [],
        #       ....
        #       'score n': [],
        # }
        self.uri_dict = self.get_uri_dict()

    def get_uri_dict(self):
        uri_dict = {}

        for user in range(self.user_nums):

            uri_dict[user] = dict()

            for rating in self.rating_list:
                item_list = self.df_data[(self.df_data['user'] == user) & (self.df_data['rating'] == rating)][
                    'item'].values
                # 不为0才加入
                if len(item_list) > 0:
                    uri_dict[user][rating] = item_list.tolist()

        return uri_dict

    def train(self, gamma=0.01, T=50, Lambda=0.01):

        for t in range(T):

            index_list = np.random.permutation(len(self.data))

            loss_rmse, loss_mae = 0, 0

            for i in index_list:
                user, item, rating = self.data[i, :]

                U_mpc = np.zeros(self.d)

                user_item_dict = self.uri_dict[user]

                for tmp_rating in user_item_dict.keys():
                    # 使用copy方法进行移除，否则会将字典中的一起移除
                    item_list = user_item_dict[tmp_rating].copy()

                    # 除去物品i的情况
                    if tmp_rating == rating:
                        item_list.remove(item)

                    # 要注意除去后长度是否为0
                    if len(item_list) > 0:
                        # 括号后面的那一部分
                        sum_M_r = np.sum(self.M[item_list], axis=0)
                        U_mpc += sum_M_r / np.sqrt(len(item_list))

                pred = self.mu + self.b_u[user] + self.b_i[item] + np.dot(self.V[item], self.U[user] + U_mpc)

                if pred > 5:
                    pred = 5
                elif pred < 1:
                    pred = 1

                e = rating - pred

                self.mu = self.mu + gamma * e
                self.b_u[user] = self.b_u[user] + gamma * (e - Lambda * self.b_u[user])
                self.b_i[item] = self.b_i[item] + gamma * (e - Lambda * self.b_i[item])
                self.U[user] = self.U[user] + gamma * (e * self.V[item] - Lambda * self.U[user])
                self.V[item] = self.V[item] + gamma * (e * (self.U[user] + U_mpc) - Lambda * self.V[item])

                # 更新M
                for tmp_rating in user_item_dict.keys():
                    # 同上
                    item_list = user_item_dict[tmp_rating].copy()

                    # 去除物品i
                    if tmp_rating == rating:
                        item_list.remove(item)

                    # 这里不用担心，因为如果列表为空，则直接跳过了
                    for i_prime in item_list:
                        self.M[i_prime] = self.M[i_prime] + gamma * (
                                    ((e * self.V[item]) / np.sqrt(len(item_list))) - (Lambda * self.M[i_prime]))

                loss_rmse += (rating - pred) ** 2
                loss_mae += abs(rating - pred)

            print(
                f'epoch: {t + 1}, rmse loss: {np.sqrt(loss_rmse / len(self.data))}, mae loss: {loss_mae / len(self.data)}')
            gamma = gamma * 0.9

    def test(self, data):
        data = np.array(data)

        loss_rmse, loss_mae = 0, 0

        for i in range(len(data)):
            user, item, rating = data[i, :]

            U_mpc = np.zeros(self.d)

            user_item_dict = self.uri_dict[user]

            for tmp_rating in user_item_dict.keys():
                item_list = self.uri_dict[user][tmp_rating]

                # 除去物品i的情况
                if tmp_rating == rating:
                    item_list.remove(item)

                # 括号后面的那一部分
                if len(item_list) > 0:
                    # 括号后面的那一部分
                    sum_M_r = np.sum(self.M[item_list], axis=0)
                    U_mpc += sum_M_r / np.sqrt(len(item_list))

            pred = self.mu + self.b_u[user] + self.b_i[item] + np.dot(self.V[item], self.U[user] + U_mpc)

            if pred > 5:
                pred = 5
            elif pred < 1:
                pred = 1

            loss_rmse += (rating - pred) ** 2
            loss_mae += abs(rating - pred)

        print(f'test result, rmse loss: {np.sqrt(loss_rmse / len(self.data))}, mae loss: {loss_mae / len(self.data)}')
