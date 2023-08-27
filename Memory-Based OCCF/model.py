import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances


class ItemCF:
    def __init__(self, user_nums, item_nums, K):
        self.user_nums = user_nums
        self.item_nums = item_nums
        self.K = K

        # 相似度矩阵
        self.item_sim_mat = None
        # 记录训练集中用户购买过的物品
        # 在进行推荐时会用到
        self.u_i_dict = {}

    def get_Nj(self, j):
        # 获得物品j和其他物品的相似度
        sim_seires = self.item_sim_mat.iloc[j, :]
        # 根据相似度从大到小排序
        sim_seires.sort_values(ascending=False, inplace=True)
        # 除去自身
        sim_seires = sim_seires.iloc[1:]

        return list(sim_seires.iloc[:self.K].index)

    def get_Iu(self, user):

        if user in self.u_i_dict.keys():
            return self.u_i_dict[user]
        else:
            return list()

    def train(self, data):

        # 交互矩阵
        u_i_mat = pd.DataFrame(np.zeros((self.user_nums, self.item_nums)), index=np.arange(0, self.user_nums),
                               columns=np.arange(0, self.item_nums))

        for i in range(len(data)):
            user, item, rating = data[i, :]
            u_i_mat.iloc[user, item] = 1

            if user not in self.u_i_dict.keys():
                self.u_i_dict[user] = [item]
            else:
                self.u_i_dict[user].append(item)

        # 自己写也很简单，但是计算速度太慢了，使用skleran中自带的包
        item_sim_mat = 1 - pairwise_distances(u_i_mat.T.values, metric='jaccard')
        self.item_sim_mat = pd.DataFrame(item_sim_mat, index=np.arange(0, self.item_nums),
                                         columns=np.arange(0, self.item_nums))

        # normalization
        for index, row in self.item_sim_mat.iterrows():
            # 最大值为1，即自身与自身的相似度
            second_large = sorted(row, reverse=True)[1]
            self.item_sim_mat.loc[index] = row / second_large

    def test(self, data):

        ranking, ground_truth = {}, {}
        total_item_list = [i for i in range(self.item_nums)]

        for i in range(len(data)):
            user, item, rating = data[i, :]

            if user not in ground_truth.keys():
                ground_truth[user] = [item]
            else:
                ground_truth[user].append(item)

            if user not in ranking.keys():
                # 对用户u进行推荐
                # 推荐的物品要从该用户未交互过的物品中进行推荐
                uniteract_item = list(set(total_item_list) - set(self.u_i_dict[user]))

                ranking_df = pd.DataFrame(index=np.arange(0, len(uniteract_item)), columns=['item', 'rating'])

                # 对每个物品选择topk物品进行评分预测，将评分作为排序的依据，从大到小进行排序
                for index, j in enumerate(uniteract_item):
                    Nj = self.get_Nj(j)
                    Iu = self.get_Iu(user)

                    # 求出两个的交集
                    K = list(set(Nj).intersection(Iu))

                    pred = 0
                    for k in K:
                        pred += self.item_sim_mat.iloc[k, j]

                    ranking_df.iloc[index, 0] = j
                    ranking_df.iloc[index, 1] = pred

                    ranking_df.sort_values(by='rating', ascending=False, inplace=True)

                ranking[user] = ranking_df['item'].tolist()

        return ranking, ground_truth


class UserCF:
    def __init__(self, user_nums, item_nums, K):
        self.user_nums = user_nums
        self.item_nums = item_nums
        self.K = K

        # 相似度矩阵
        self.user_sim_mat = None
        self.u_i_dict = {}
        self.i_u_dict = {}

    # 获得对物品j交互过的用户
    def get_Uj(self, j):
        if j in self.i_u_dict:
            return self.i_u_dict[j]
        else:
            return list()

    # 获得用户的K近邻
    def get_Nu(self, user):
        sim_series = self.user_sim_mat.iloc[user]
        sim_series.sort_values(ascending=False, inplace=True)
        sim_series = sim_series.iloc[1:]

        return list(sim_series.iloc[:self.K].index)

    def train(self, data):
        # 交互矩阵
        u_i_mat = pd.DataFrame(np.zeros((self.user_nums, self.item_nums)), index=np.arange(0, self.user_nums),
                               columns=np.arange(0, self.item_nums))

        for i in range(len(data)):
            user, item, rating = data[i, :]
            u_i_mat.iloc[user, item] = 1

            if user not in self.u_i_dict.keys():
                self.u_i_dict[user] = [item]
            else:
                self.u_i_dict[user].append(item)

            if item not in self.i_u_dict.keys():
                self.i_u_dict[item] = [user]
            else:
                self.i_u_dict[item].append(user)

        user_sim_mat = 1 - pairwise_distances(u_i_mat.values, metric='jaccard')
        self.user_sim_mat = pd.DataFrame(user_sim_mat, index=np.arange(0, self.user_nums),
                                         columns=np.arange(0, self.user_nums))

        # normalization
        for index, row in self.user_sim_mat.iterrows():
            # 最大值为1，即自身与自身的相似度
            second_large = sorted(row, reverse=True)[1]
            self.user_sim_mat.loc[index] = row / second_large

    def test(self, data):
        ranking, ground_truth = {}, {}
        total_item_list = [i for i in range(self.item_nums)]

        for i in range(len(data)):
            user, item, rating = data[i, :]

            if user not in ground_truth.keys():
                ground_truth[user] = [item]
            else:
                ground_truth[user].append(item)

            if user not in ranking.keys():
                uninteract_item_list = list(set(total_item_list) - set(self.u_i_dict[user]))

                ranking_df = pd.DataFrame(index=np.arange(0, len(uninteract_item_list)), columns=['item', 'rating'])

                for index, j in enumerate(uninteract_item_list):
                    Uj = self.get_Uj(j)
                    Nu = self.get_Nu(user)

                    W = list(set(Uj).intersection(set(Nu)))

                    pred = 0
                    for w in W:
                        pred += self.user_sim_mat.iloc[w, user]

                    ranking_df.iloc[index, :] = j, pred

                ranking_df.sort_values(by='rating', ascending=False, inplace=True)

                ranking[user] = ranking_df['item'].tolist()

        return ranking, ground_truth
