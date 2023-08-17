import pandas as pd
import numpy as np

# 设置numpy的精确位数
np.set_printoptions(precision = 4)


class AF:
    def __init__(self, data):
        self.train_user, self.train_item = list(set(data[:, 0])), list(set(data[:, 1]))
        self.data = pd.DataFrame(data, columns=['user', 'item', 'rating'])

    def train(self):
        self.avg_r = self.data.loc[:, 'rating'].mean()

        self.avg_r_u, self.avg_r_i, self.b_u, self.b_i = {}, {}, {}, {}

        # 对于每个用户计算 平均r_u
        for user in self.train_user:
            mask = self.data.loc[:, 'user'] == user
            data = self.data.loc[mask]
            self.avg_r_u[user] = data.loc[:, 'rating'].mean()

        # 对于每个物品计算 平均r_i
        for item in self.train_item:
            mask = self.data.loc[:, 'item'] == item
            data = self.data.loc[mask]
            self.avg_r_i[item] = data.loc[:, 'rating'].mean()

        # 对于每个用户计算 b_u
        for user in self.train_user:
            mask = self.data.loc[:, 'user'] == user
            data = self.data.loc[mask]

            bias_sum = 0

            for row in data.itertuples():
                item, rating = row[2], row[3]
                bias_sum += rating - self.avg_r_i[item]

            self.b_u[user] = bias_sum / len(data)

        # 对于每个物品计算 b_i
        for item in self.train_item:
            mask = self.data.loc[:, 'item'] == item
            data = self.data.loc[mask]

            bias_sum = 0

            for row in data.itertuples():
                user, rating = row[1], row[3]
                bias_sum += rating - self.avg_r_u[user]

            self.b_i[item] = bias_sum / len(data)

    def test(self, data):

        total_loss_rmse, total_loss_mae = [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]

        # 同时计算六种方式的预测值
        for i in range(len(data)):
            user, item, rating = data[i, :]

            # 预测规则
            # 如果测试集的用户不在训练集中，则avg_r_u=avg_r, b_u=0
            # 如果测试集的物品不在训练集中，则avg_r_i=avg_r, b_i=0

            # 整理要用到的参数
            if user in self.avg_r_u.keys():
                avg_r_u = self.avg_r_u[user]
            else:
                avg_r_u = self.avg_r

            if item in self.avg_r_i.keys():
                avg_r_i = self.avg_r_i[item]
            else:
                avg_r_i = self.avg_r

            if user in self.b_u.keys():
                b_u = self.b_u[user]
            else:
                b_u = 0

            if item in self.b_i.keys():
                b_i = self.b_i[item]
            else:
                b_i = 0


            # 利用平均r_u
            pred = avg_r_u
            total_loss_rmse[0] += (rating - pred) ** 2
            total_loss_mae[0] += abs(rating - pred)

            ########################################

            # 利用平均r_i
            pred = avg_r_i
            total_loss_rmse[1] += (rating - pred) ** 2
            total_loss_mae[1] += abs(rating - pred)

            ########################################

            # 利用 avg_r_u 和 avg_r_i
            pred = 0.5 * avg_r_u + 0.5 * avg_r_i
            total_loss_rmse[2] += (rating - pred) ** 2
            total_loss_mae[2] += abs(rating - pred)

            ########################################

            # 利用b_u和avg_r_i
            pred = b_u + avg_r_i
            total_loss_rmse[3] += (rating - pred) ** 2
            total_loss_mae[3] += abs(rating - pred)

            ########################################

            # 利用b_i和avg_r_u
            pred = b_i + avg_r_u
            total_loss_rmse[4] += (rating - pred) ** 2
            total_loss_mae[4] += abs(rating - pred)

            ########################################

            # 利用avg_r和b_u和b_i
            pred = self.avg_r + b_u + b_i
            total_loss_rmse[5] += (rating - pred) ** 2
            total_loss_mae[5] += abs(rating - pred)

        print(f"rmse_loss: {np.sqrt(np.array(total_loss_rmse) / len(data))}")
        print(f"mae_loss: {np.array(total_loss_mae) / len(data)}")