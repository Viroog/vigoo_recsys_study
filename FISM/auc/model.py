import numpy as np
from collections import defaultdict
from FISM.evaluation import  Evaluation


class FISMauc:
    def __init__(self, dataloader, alpha, gamma, Au_size, alpha_v, alpha_w, beta_v, d, T):
        self.dataloader = dataloader
        self.alpha = alpha
        self.gamma = gamma
        self.Au_size = Au_size
        self.alpha_v = alpha_v
        self.alpha_w = alpha_w
        self.beta_v = beta_v
        self.d = d
        self.T = T

        self.I = [i for i in range(1, self.dataloader.max_i + 1)]

        # 这里没有b_u，因为在做pair-wise的时候会被消掉
        self.b_i = np.random.randn(self.dataloader.max_i + 1) * 0.01
        self.W = np.random.randn(self.dataloader.max_i + 1, self.d) * 0.01
        self.V = np.random.randn(self.dataloader.max_i + 1, self.d) * 0.01

    def train(self):
        for t in range(self.T):
            # 根据auc的算法，取的三元组是(u,i,Au)，其中Au是采样的unobserved item

            # 采用的是user-wise，先遍历每个用户，再遍历每个物品。而不是直接取(u,i)对
            # 收敛更慢，但效果更好
            for u, Iu in self.dataloader.train_u_i_dict.items():
                # 先准备好不用重复计算
                I_min_Iu = list(set(self.I) - set(Iu))
                n_u = len(Iu)

                for i in Iu:
                    # 会存在一个用户u只交互过一个物品的情况，此时分母为0，要特殊处理
                    if n_u == 1:
                        # 赋值任意值，因为Uu_min_u为0
                        normed_factor1 = 0
                    else:
                        normed_factor1 = (n_u - 1) ** -self.alpha
                    # 计算物品i的用户肖像向量
                    Uu_min_i = np.zeros(self.d)
                    for i_pi in Iu:
                        if i_pi != i:
                            Uu_min_i += self.W[i_pi]
                    # normalization
                    Uu_min_i *= normed_factor1

                    normed_factor2 = n_u ** -self.alpha
                    # 计算物品j的用户肖像向量
                    Uu = np.zeros(self.d)
                    for i_pi in Iu:
                        Uu += self.W[i_pi]
                    # normalization
                    Uu *= normed_factor2

                    # 从I_min_Iu随机抽取出Au_size个比赛
                    Au = np.random.choice(I_min_Iu, size=self.Au_size, replace=False)
                    # Au = I_min_Iu[random_index]
                    # 求四个累积量
                    # 其中x1,x2是vector，x3,x4是scaler
                    x1, x2, x3, x4 = np.zeros(self.d), np.zeros(self.d), 0, 0

                    r_ui = self.b_i[i] + np.dot(Uu_min_i, self.V[i])

                    # 计算累积量，并在每一次计算中更新bj和Vj
                    for j in Au:
                        r_uj = self.b_i[j] + np.dot(Uu, self.V[j])
                        # 这个损失的含义是，让预测的r_ui和r_uj的距离(差值)尽量保持为1
                        e_uij = (1 - (r_ui - r_uj)) / self.Au_size

                        # W_i_pi
                        x1 += -e_uij * (normed_factor1 * self.V[i] - normed_factor2 * self.V[j])
                        # W_i
                        x2 += e_uij * normed_factor2 * self.V[j]
                        # b_i
                        x3 += -e_uij
                        # V_i
                        x4 += -e_uij * Uu_min_i

                        # update bj, Vj
                        self.b_i[j] = self.b_i[j] - self.gamma * (e_uij + self.beta_v * self.b_i[j])
                        self.V[j] = self.V[j] - self.gamma * (e_uij * Uu + self.alpha_v * self.V[j])

                    self.b_i[i] = self.b_i[i] - self.gamma * (x3 + self.beta_v * self.b_i[i])
                    self.V[i] = self.V[i] - self.gamma * (x4 + self.alpha_v * self.V[i])

                    for i_pi in Iu:
                        if i_pi != i:
                            self.W[i_pi] = self.W[i_pi] - self.gamma * (x1 + self.alpha_w * self.W[i_pi])

                    self.W[i] = self.W[i] - self.gamma * (x2 + self.alpha_w * self.W[i])

            if (t+1) % 10 == 0:
                print(f"epoch: {t+1}:")
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

                # 不需要减去i
                Uu = np.zeros(self.d)

                for i_pi in test_u_item:
                    if i_pi != i:
                        Uu += self.W[i_pi]

                Uu *= normed_factor

                r_hat = self.b_i[i] + np.dot(Uu, self.V[i])

                u_ranking_item[i] = r_hat

            # 排序
            sorted_u_ranking_item = dict(sorted(u_ranking_item.items(), key=lambda x: x[1], reverse=True))
            ranking_u_i_dict[u] = list(sorted_u_ranking_item.keys())

        k = 5
        evaluation = Evaluation(k, ranking_u_i_dict, self.dataloader.test_u_i_dict)
        print(f"Pre@{k}: {evaluation.precision()}")
        print(f"Rec@{k}: {evaluation.recall()}")
