import numpy as np
from evaluation import Evaluation
from collections import defaultdict


class MFlogloss:
    def __init__(self, dataloader, lu, d, gamma, alpha_u, alpha_v, beta_u, beta_v, T):
        self.dataloader = dataloader
        self.lu = lu
        self.d = d
        self.gamma = gamma
        self.alpha_u = alpha_u
        self.alpha_v = alpha_v
        self.beta_u = beta_u
        self.beta_v = beta_v
        self.T = T

        # total item
        self.I = [i for i in range(1, self.dataloader.max_i + 1)]

        self.b_u = np.random.randn(self.dataloader.max_u + 1) * 0.01
        self.b_i = np.random.randn(self.dataloader.max_i + 1) * 0.01

        self.U = np.random.randn(self.dataloader.max_u + 1, self.d) * 0.01
        self.V = np.random.randn(self.dataloader.max_i + 1, self.d) * 0.01

        unobserved_data = []

        for u, Iu in self.dataloader.train_u_i_dict.items():
            I_min_Iu = list(set(self.I) - set(Iu))

            for j in I_min_Iu:
                # rating=-1代表为交互过
                unobserved_data.append([u, j, -1])

        self.unobserved_data = np.array(unobserved_data)

    def train(self):

        P = self.dataloader.train_data
        A_size = self.lu * len(P)

        for t in range(self.T):

            random_row = np.random.choice(self.unobserved_data.shape[0], size=A_size, replace=False)
            A = self.unobserved_data[random_row]

            total_data = np.vstack((P, A))
            np.random.shuffle(total_data)

            for index in range(len(total_data)):
                u, i, r_ui = total_data[index, :]

                r_ui_hat = self.b_u[u] + self.b_i[i] + np.dot(self.U[u], self.V[i])

                e_ui = r_ui / (1 + np.exp(r_ui * r_ui_hat))

                self.b_u[u] += self.gamma * (e_ui - self.beta_u * self.b_u[u])
                self.b_i[i] += self.gamma * (e_ui - self.beta_v * self.b_i[i])
                self.V[i] += self.gamma * (e_ui * self.U[u] - self.alpha_v * self.V[i])
                self.U[u] += self.gamma * (e_ui * self.V[i] - self.alpha_u * self.U[u])

            if (t+1) % 10 == 0:
                print(f"epoch{t+1}:")
                self.test()

    def test(self):

        ranking_u_i_dict = defaultdict(list)

        for u, _ in self.dataloader.test_u_i_dict.items():

            Iu = self.dataloader.train_u_i_dict[u]
            I_min_Iu = list(set(self.I) - set(Iu))

            u_ranking_item = {}

            for i in I_min_Iu:

                r_ui = self.b_u[u] + self.b_i[i] + np.dot(self.U[u], self.V[i])

                u_ranking_item[i] = r_ui

            # 排序
            sorted_u_ranking_item = dict(sorted(u_ranking_item.items(), key=lambda x: x[1], reverse=True))
            ranking_u_i_dict[u] = list(sorted_u_ranking_item.keys())

        k = 5
        evaluation = Evaluation(k, ranking_u_i_dict, self.dataloader.test_u_i_dict)
        print(f"Pre@{k}: {evaluation.precision()}")
        print(f"Rec@{k}: {evaluation.recall()}")


