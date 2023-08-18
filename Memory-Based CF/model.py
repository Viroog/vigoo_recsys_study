import numpy as np
import pandas as pd

# 设置小数精确到小数点后四位
np.set_printoptions(precision=4)


class UCF:
    def __init__(self, data, K=50):
        self.train_user, self.train_item = list(set(data[:, 0])), list(set(data[:, 1]))
        self.data = pd.DataFrame(data, columns=['user', 'item', 'rating'])

        # 预测时要用到 avg_r_u, 在训练过程中计算
        self.avg_r_u = {}
        # 不是所有用户都需要计算相似度，在测试的时候再计算，减少不必要的计算
        self.sim_dict = {}
        # K近邻
        self.K = K

    # 返回user交互过的物品集合
    # 返回类型为list
    def get_Item_Set(self, user):
        mask = self.data.loc[:, 'user'] == user
        data = self.data.loc[mask]

        return list(data.loc[:, 'item'])

    def cal_sim(self, u, w, K):
        # 分别对应皮尔逊相关系数的，分子，左边的分母和右边的分母的求和部分
        cov, sigma1, sigma2 = 0, 0, 0
        avg_r_u, avg_r_w = self.avg_r_u[u], self.avg_r_u[w]

        for k in K:
            bias1 = self.data[(self.data['user'] == u) & (self.data['item'] == k)]['rating'].values[0] - avg_r_u
            bias2 = self.data[(self.data['user'] == w) & (self.data['item'] == k)]['rating'].values[0] - avg_r_w

            cov += bias1 * bias2
            sigma1 += bias1 ** 2
            sigma2 += bias2 ** 2

        # 有可能出现分母为0的情况，这时候要返回0，然后再做特判
        if cov == 0 or sigma1 == 0 or sigma2 == 0:
            return 0

        return cov / (np.sqrt(sigma1) * np.sqrt(sigma2))

    # Uj是评价过物品j的用户
    def get_Uj(self, item):
        mask = self.data.loc[:, 'item'] == item
        data = self.data.loc[mask]

        return list(data.loc[:, 'user'])

    # 获得用户u的邻居(s_wu不为0的用户)
    def get_Nu(self, user):
        return self.sim_dict[user]

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
            sorted_items = sorted(result.items(), key=lambda item: item[1], reverse=True)
            top_n_items = sorted_items[:self.K]
            top_n_dict = dict(top_n_items)

            return top_n_dict

    def train(self):

        for user in self.train_user:
            mask = self.data.loc[:, 'user'] == user
            data = self.data.loc[mask]

            self.avg_r_u[user] = data.loc[:, 'rating'].mean()

    def test(self, data):

        loss_rmse, loss_mae = 0, 0
        result = []

        for i in range(len(data)):
            u, item, rating = data[i, :]

            # 用户id不在sim_dict才用计算，因为第一次算就把他和其他所有用户算过了
            if u not in self.sim_dict.keys():

                # Iu为用户u评过分的物品集合
                Iu = self.get_Item_Set(u)

                # 计算每一个用户w的相似度
                for w in self.train_user:
                    if w != u:
                        # Iw为用户w评过分的物品集合
                        Iw = self.get_Item_Set(w)
                        K = list(set(Iu).intersection(set(Iw)))

                        # 如果交集K为0，说明两个用户完全不相似，不需要考虑
                        if len(K) > 0:
                            sim = self.cal_sim(u, w, K)

                            # 返回有可能会出现0的情况，要特判
                            if sim != 0:
                                # 第一次是创建字典，第二次则是往字典插值
                                if u not in self.sim_dict.keys():
                                    self.sim_dict[u] = {w: sim}
                                else:
                                    self.sim_dict[u][w] = sim

            # Nu为字典, Uj为列表
            Nu, Uj = self.get_Nu(u), self.get_Uj(item)
            # W是一个字典
            W = self.get_topk(Nu, Uj)

            avg_r_u = self.avg_r_u[u]
            # 分子、分母
            numerator, denominator = 0, 0

            for w, sim in W.items():
                r_wj, avg_r_w = self.data[(self.data['user'] == w) & (self.data['item'] == item)]['rating'].values[0], \
                    self.avg_r_u[w]

                numerator += sim * (r_wj - avg_r_w)
                denominator += abs(sim)

            # 分母为0的情况，即空集
            if denominator == 0:
                pred = avg_r_u
            else:
                pred = avg_r_u + numerator / denominator

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
    def __init__(self, data):
        pass

    def train(self):
        pass

    def test(self, data):
        pass


class HCF:
    def __init__(self, pred_ucf, pred_icf, Lambda=0.5):
        self.pred_ucf = pred_ucf
        self.pred_icf = pred_icf
        self.Lambda = Lambda

    def test(self, data):
        pred = self.Lambda * self.pred_ucf + (1 - self.Lambda) * self.pred_ucf
        rating = data[:, 2]

        loss_rmse = np.sqrt(((rating - pred) ** 2).sum() / len(data))
        loss_mae = np.abs(rating - pred).sum() / len(data)

        print(f"User-Based CF, RMSE: {loss_rmse}, MAE: {loss_mae}")