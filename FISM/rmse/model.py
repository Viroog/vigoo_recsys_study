import numpy as np
from collections import defaultdict
from FISM.evaluation import Evaluation


class FISMrmse:
    def __init__(self, dataloader, alpha, gamma, lu, d, alpha_v, alpha_w, beta_u, beta_v, T):
        self.dataloader = dataloader
        self.alpha = alpha
        self.gamma = gamma
        self.lu = lu
        self.d = d
        self.alpha_v = alpha_v
        self.alpha_w = alpha_w
        self.beta_u = beta_u
        self.beta_v = beta_v
        self.T = T

        self.W = np.random.randn(self.dataloader.max_i + 1, self.d) * 0.01
        self.V = np.random.randn(self.dataloader.max_i + 1, self.d) * 0.01

        self.b_i = np.random.randn(self.dataloader.max_i + 1) * 0.01
        self.b_u = np.random.randn(self.dataloader.max_u + 1) * 0.01

        # 所有物品
        self.I = [i for i in range(1, self.dataloader.max_i + 1)]

        unobserved_data = []

        # u是用户，v是该用户交互过的物品
        for u, v in self.dataloader.train_u_i_dict.items():
            unobserved_items = list(set(self.I) - set(v))
            pairs = [[u, i] for i in unobserved_items]

            unobserved_data.extend(pairs)

        # 所有未观测到的数据
        self.unobserved_data = np.array(unobserved_data)

    def train(self):
        for t in range(self.T):
            # 每个epoch都重新抽样一次
            R = self.dataloader.train_data

            # 在R的最后一列后插入全为1的列，表示交互过，后面计算损失要用到
            ones_column = np.ones((R.shape[0], 1), dtype=int)
            R = np.concatenate((R, ones_column), axis=1)

            # 随机抽样
            # 抽样集合的大小
            A_size = self.lu * len(self.dataloader.train_data)
            # replace=False可以保证一次抽样中不会抽到同一行的情况
            random_row = np.random.choice(self.unobserved_data.shape[0], size=A_size, replace=False)
            A = self.unobserved_data[random_row]

            # 同理，插入0列
            zeros_column = np.zeros((A.shape[0], 1), dtype=int)
            A = np.concatenate((A, zeros_column), axis=1)

            # R和A的并集
            R_pi = np.vstack((R, A))
            # shuffle
            np.random.shuffle(R_pi)

            for i in range(len(R_pi)):
                u, i, r = R_pi[i, :]

                # 这里计算x是有问题的

                n_u = len(self.dataloader.train_u_i_dict[u])
                # 物品i是被观测到的，正则化系数要减一，再开次方
                if r == 1:
                    # 如果n_u为1了则不能减一，x也为0向量，正则化系数为多少无所谓，默认为0
                    if n_u == 1:
                        normed_factor = 0
                    else:
                        normed_factor = (n_u - 1) ** -self.alpha
                # 物品i是未被观测到的，正则化系数则不用减一，直接开放
                else:
                    normed_factor = n_u ** -self.alpha

                # 先计算x，初始化为0
                x = np.zeros(self.d)
                for j in self.dataloader.train_u_i_dict[u]:
                    # 显示排除i
                    if j != i:
                        x += self.W[j]

                x *= normed_factor

                # 预测值
                r_hat = self.b_u[u] + self.b_i[i] + np.dot(x, self.V[i])
                # 误差
                e = r - r_hat

                self.b_u[u] += self.gamma * (e - self.beta_u * self.b_u[u])
                self.b_i[i] += self.gamma * (e - self.beta_v * self.b_i[i])
                self.V[i] += self.gamma * (e * x - self.alpha_v * self.V[i])

                for j in self.dataloader.train_u_i_dict[u]:
                    if j != i:
                        self.W[j] += self.gamma * (e * normed_factor * self.V[i] - self.alpha_w * self.W[j])

            if (t + 1) % 10 == 0:
                print(f"epoch {t + 1}:")
                self.test()

    def test(self):

        ranking_u_i_dict = defaultdict(list)

        # 对于每个用户预测其未购买过的物品的评分
        for u, test_u_item in self.dataloader.test_u_i_dict.items():
            Iu = self.dataloader.train_u_i_dict[u]
            # 未购买过的物品
            I_min_Iu = set(self.I) - set(Iu)

            u_ranking_item = {}

            n_u = len(test_u_item)
            # 这里都是对未观测过的物品进行评分预测，因此正则化系数不需要减一可以直接次方
            normed_factor = n_u ** -self.alpha

            # 对未在训练集中的物品预测评分
            for i in I_min_Iu:

                x = np.zeros(self.d)

                for j in test_u_item:
                    if j != i:
                        x += self.W[j]

                x *= normed_factor

                r_hat = self.b_u[u] + self.b_i[i] + np.dot(x, self.V[i])

                u_ranking_item[i] = r_hat

            # 排序
            sorted_u_ranking_item = dict(sorted(u_ranking_item.items(), key=lambda x: x[1], reverse=True))
            ranking_u_i_dict[u] = list(sorted_u_ranking_item.keys())

        k = 5
        evaluation = Evaluation(k, ranking_u_i_dict, self.dataloader.test_u_i_dict)
        print(f"Pre@{k}: {evaluation.precision()}")
        print(f"Rec@{k}: {evaluation.recall()}")