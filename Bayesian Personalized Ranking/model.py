import random
from collections import defaultdict

import numpy as np

from evaluation import Evaluation


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class BPR:
    def __init__(self, dataloader, gamma, alpha_u, alpha_v, beta_v, T, d):
        self.dataloader = dataloader
        self.gamma = gamma
        self.alpha_u = alpha_u
        # alpha是针对于V的，beta是针对于b的
        self.alpha_v = alpha_v
        self.beta_v = beta_v
        self.T = T
        self.d = d

        self.I = [i for i in range(1, self.dataloader.max_i + 1)]

        # 只用到了这三个变量，没有了mu和b_u，因为在做r_ui-r_uj的时候这两项被消除了
        self.U = np.random.randn(self.dataloader.max_u + 1, self.d) * 0.01
        self.V = np.random.randn(self.dataloader.max_i + 1, self.d) * 0.01
        self.b_i = np.random.randn(self.dataloader.max_i + 1) * 0.01

        # #统计方法的初始化
        # self.U = np.zeros(shape=(self.dataloader.max_u + 1, self.d))
        # for u in range(self.dataloader.max_u+1):
        #     for k in range(self.d):
        #         self.U[u][k] = (random.random() - 0.5) * 0.01
        # self.V = np.zeros(shape=(self.dataloader.max_i + 1, self.d))
        # for i in range(self.dataloader.max_i+1):
        #     for k in range(self.d):
        #         self.V[i][k] = (random.random() - 0.5) * 0.01
        # 物品偏置量
        # mu = len(self.dataloader.train_data) / self.dataloader.max_i / self.dataloader.max_u
        # self.b_i = np.zeros(self.dataloader.max_i + 1)
        # for i in range(1, self.dataloader.max_i + 1):
        # self.b_i[i] = len(self.dataloader.train_i_u_dict[i]) / self.dataloader.max_u - mu

    def train(self):
        print("Start train!")

        for t in range(self.T):
            for t2 in range(len(self.dataloader.train_data)):
                # 第一种抽样方法：先随机抽取一个(u,i)对，然后再抽取j
                # np.random.randint生成一个随机整数，从0到shape[0]-1
                random_row_index = np.random.randint(self.dataloader.train_data.shape[0])
                random_row = self.dataloader.train_data[random_row_index]

                u, i = random_row

                # j为I/Iu中的一个随机物品
                I_min_Iu = set(self.I) - set(self.dataloader.train_u_i_dict[u])
                j = random.choice(list(I_min_Iu))

                # # 第二种抽样方法：先随机抽一个u，然后抽取i，最后抽取j
                # u = random.choice(list(self.dataloader.train_u_i_dict.keys()))
                # Iu = self.dataloader.train_u_i_dict[u]
                # i = random.choice(Iu)
                #
                # I_min_Iu = set(self.I) - set(Iu)
                # j = random.choice(list(I_min_Iu))

                r_ui, r_uj = np.dot(self.U[u], self.V[i].T) + self.b_i[i], np.dot(self.U[u], self.V[j].T) + self.b_i[j]
                r_uij = r_ui - r_uj

                sigmoid = 1.0 / (1 + np.exp(r_uij))

                self.U[u] = self.U[u] - self.gamma * (-sigmoid * (self.V[i] - self.V[j]) + self.alpha_u * self.U[u])
                self.V[i] = self.V[i] - self.gamma * (-sigmoid * self.U[u] + self.alpha_v * self.V[i])
                self.V[j] = self.V[j] - self.gamma * (-sigmoid * (-self.U[u]) + self.alpha_v * self.V[j])
                self.b_i[i] = self.b_i[i] - self.gamma * (-sigmoid + self.beta_v * self.b_i[i])
                self.b_i[j] = self.b_i[j] - self.gamma * (-sigmoid * (-1) + self.beta_v * self.b_i[j])

            if (t + 1) % 10 == 0:
                print(f"epoch: {t + 1}:")
                # 每隔10个epoch测试一次
                self.test()

    def test(self):
        # 每个用户的推荐序列，列表顺序为预测评分高的优先
        ranking_u_i_dict = defaultdict(list)

        # 对于测试集的每一个用户
        for u, _ in self.dataloader.test_u_i_dict.items():
            Iu = self.dataloader.train_u_i_dict[u]
            I_min_Iu = set(self.I) - set(Iu)

            # 该用户未购买过的物品的预测评分
            u_ranking_item = {}

            # 遍历每个未购买过的物品，预测评分
            for i in I_min_Iu:
                r_ui = np.dot(self.U[u], self.V[i].T) + self.b_i[i]

                u_ranking_item[i] = r_ui

            # 对item根据评分进行从大到小排序
            sorted_u_ranking_item = dict(sorted(u_ranking_item.items(), key=lambda x: x[1], reverse=True))

            ranking_u_i_dict[u] = list(sorted_u_ranking_item.keys())

        k = 5
        evaluation = Evaluation(k, ranking_u_i_dict, self.dataloader.test_u_i_dict)
        print(f"Pre@{k}: {evaluation.precision()}")
        print(f"Rec@{k}: {evaluation.recall()}")
