import numpy as np
import pandas as pd

# 设置小数精确到小数点后四位
np.set_printoptions(precision=4)


class UCF:
    def __init__(self, data, K=50):

        # 原来的数据类型为np.array, 现在转为dataframe
        self.data = pd.DataFrame(data, columns=['user', 'item', 'rating'])

        min_user, max_user = data[:, 0].min(), data[:, 0].max()
        min_item, max_item = data[:, 1].min(), data[:, 1].max()

        row_index = [i for i in range(min_user, max_user + 1)]
        col_index = [j for j in range(min_item, max_item + 1)]

        # 用户-物品交互矩阵
        self.interaction = pd.DataFrame(index=row_index, columns=col_index)

        for i in range(len(data)):
            user, item, rating = data[i, :]
            self.interaction.loc[user, item] = rating

        self.train_user, self.train_item = list(set(data[:, 0])), list(set(data[:, 1]))
        # 预测时要用到 avg_r_u, 在训练过程中计算
        self.avg_r_u = {}
        # K近邻
        self.K = K
        print('UCF Model start!')

    # 返回user交互过的物品集合
    # 返回类型为list
    def get_Item_Set(self, user):
        mask = self.data.loc[:, 'user'] == user
        data = self.data.loc[mask]

        return list(data.loc[:, 'item'])

    # Uj是评价过物品j的用户
    def get_Uj(self, item):
        mask = self.data.loc[:, 'item'] == item
        data = self.data.loc[mask]

        return list(data.loc[:, 'user'])

    # 获得用户u的邻居(s_wu不为0的用户)
    def get_Nu(self, user):
        # 该用户与其他用户的一个皮尔逊系数，类型为series
        u_correlation = self.correlation_matrix.loc[user, :]
        u_correlation = u_correlation.dropna()
        # 去除相似度为0的数据
        mask = u_correlation != 0
        u_correlation = u_correlation.loc[mask]

        # 转成字典
        Nu = u_correlation.to_dict()
        # 去除该用户自身与自身的皮尔逊系数
        del Nu[user]
        return Nu

    # 如果交集不满足k个或刚好为k个，则选全部
    # 若交集大于k个，则取最相似的k个
    def get_topk(self, Nu, Uj):
        intersection = list(set(Nu.keys()).intersection(set(Uj)))

        result = {}

        for user in intersection:
            result[user] = Nu[user]

        if len(intersection) <= self.K:
            return result
        else:
            sorted_users = sorted(result.items(), key=lambda item: item[1], reverse=True)
            top_n_users = sorted_users[:self.K]
            top_n_dict = dict(top_n_users)

            return top_n_dict

    def train(self):
        for user in self.train_user:
            mask = self.data.loc[:, 'user'] == user
            data = self.data.loc[mask]

            self.avg_r_u[user] = data.loc[:, 'rating'].mean()

        self.correlation_matrix = self.interaction.T.corr(numeric_only=False)

    def test(self, data):

        loss_rmse, loss_mae = 0, 0
        result = []

        for i in range(len(data)):
            user, item, rating = data[i, :]

            # Nu为字典, Uj为列表
            Nu, Uj = self.get_Nu(user), self.get_Uj(item)

            # W是一个字典
            W = self.get_topk(Nu, Uj)

            avg_r_u = self.avg_r_u[user]

            if len(W) == 0:
                pred = avg_r_u
            else:
                # 分子、分母
                numerator, denominator = 0, 0

                for w, sim in W.items():
                    r_wj, avg_r_w = self.data[(self.data['user'] == w) & (self.data['item'] == item)]['rating'].values[0], \
                        self.avg_r_u[w]

                    numerator += sim * (r_wj - avg_r_w)
                    # 两种计算方式
                    # denominator += sim
                    denominator += abs(sim)

                pred = avg_r_u + numerator / denominator

                # 计算出来的预测值可能会大于5或小于1，特殊处理
                if pred > 5:
                    pred = 5
                elif pred < 1:
                    pred = 1

            result.append(pred)

            loss_rmse += (rating - pred) ** 2
            loss_mae += abs(rating - pred)

        print(f"User-Based CF, RMSE: {np.sqrt(loss_rmse / len(data))}, MAE: {loss_mae / len(data)}")

        return np.array(result)


class ICF:
    def __init__(self, data, K=50):
        # 原来的数据类型为np.array, 现在转为dataframe
        self.data = pd.DataFrame(data, columns=['user', 'item', 'rating'])

        min_user, max_user = data[:, 0].min(), data[:, 0].max()
        min_item, max_item = data[:, 1].min(), data[:, 1].max()

        row_index = [i for i in range(min_user, max_user + 1)]
        col_index = [j for j in range(min_item, max_item + 1)]

        # 用户-物品交互矩阵
        self.interaction = pd.DataFrame(index=row_index, columns=col_index)

        for i in range(len(data)):
            user, item, rating = data[i, :]
            self.interaction.loc[user, item] = rating

        self.train_user, self.train_item = list(set(data[:, 0])), list(set(data[:, 1]))
        # 预测时要用到 avg_r_u, 在训练过程中计算
        self.avg_r_u = {}
        # K近邻
        self.K = K
        print('ICF Model start!')

    # Nj为字典，键为物品j的邻居，值为该邻居与物品j的相似度
    def get_Nj(self, item):
        i_correlation = self.correlation_matrix[item]
        i_correlation = i_correlation.dropna()

        mask = i_correlation != 0
        i_correlation = i_correlation.loc[mask]

        Nj = i_correlation.to_dict()

        # 有可能会出现Nj为零的情况，即在训练集中，没有用户购买过这个物品
        if len(Nj) == 0:
            return dict()

        del Nj[item]
        return Nj

    # Iu为列表，用于u交互过的物品集合
    def get_Iu(self, user):
        mask = self.data.loc[:, 'user'] == user
        data = self.data.loc[mask]

        return list(data.loc[:, 'item'])

    def get_topk(self, Nj, Iu):
        intersection = list(set(Nj.keys()).intersection(set(Iu)))

        result = {}

        for item in intersection:
            result[item] = Nj[item]

        if len(intersection) <= self.K:
            return result
        else:
            sorted_items = sorted(result.items(), key=lambda item: item[1], reverse=True)
            top_n_items = sorted_items[:self.K]
            top_n_dict = dict(top_n_items)

            return top_n_dict

    def train(self):
        for user in self.train_user:
            mask = self.data.loc[:, 'user'] == user
            data = self.data.loc[mask]

            self.avg_r_u[user] = data.loc[:, 'rating'].mean()

        self.correlation_matrix = self.interaction.corr(numeric_only=False)

    def test(self, data):

        loss_rmse, loss_mae = 0, 0
        result = []

        for i in range(len(data)):
            user, item, rating = data[i, :]

            Nj, Iu = self.get_Nj(item), self.get_Iu(user)

            W = self.get_topk(Nj, Iu)

            avg_r_u = self.avg_r_u[user]

            if len(W) == 0:
                pred = avg_r_u
            else:
                numerator, denominator = 0, 0

                for w, sim in W.items():
                    r_uk = self.data[(self.data['user'] == user) & (self.data['item'] == w)]['rating'].values[0]

                    numerator += sim * r_uk
                    # 两种计算方式
                    denominator += sim

                pred = numerator / denominator

                if pred > 5:
                    pred = 5
                elif pred < 1:
                    pred = 1

            result.append(pred)

            loss_rmse += (rating - pred) ** 2
            loss_mae += abs(rating - pred)

        print(f"Item-Based CF, RMSE: {np.sqrt(loss_rmse / len(data))}, MAE: {loss_mae / len(data)}")

        return np.array(result)


class HCF:
    def __init__(self, Lambda=0.5):
        self.pred_ucf = np.loadtxt('./pred_ucf.txt')
        self.pred_icf = np.loadtxt('./pred_icf.txt')
        self.Lambda = Lambda

    def test(self, data):
        pred = self.Lambda * self.pred_ucf + (1 - self.Lambda) * self.pred_icf
        rating = data[:, 2]

        loss_rmse = np.sqrt(((rating - pred) ** 2).sum() / len(data))
        loss_mae = np.abs(rating - pred).sum() / len(data)

        print(f"Hybrid CF, RMSE: {loss_rmse}, MAE: {loss_mae}")
