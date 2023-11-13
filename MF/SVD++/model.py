import numpy as np


# 随机数种子
# np.random.seed(42)


class SVDpp:
    def __init__(self, data, k=20):
        # training Data
        self.data = np.array(data)
        # latent dim，即用户向量和物品向量的维度
        self.k = k

        # num of user & items
        self.user_nums = len(set(self.data[:, 0]))
        self.item_nums = len(set(self.data[:, 1]))

        # b_ui = μ + b_u + b_i
        # μ:average of ratings  b_u:user bias  b_i:item bias
        self.mu = self.data[:, 2].mean()
        # 两种初始化方法 1.0初始化 2.正态分布初始化
        # self.b_u = np.zeros(self.user_nums, dtype=np.float)
        self.b_u = np.random.randn(self.user_nums)
        # self.b_i = np.zeros(self.item_nums, dtype=np.float)
        self.b_i = np.random.randn(self.item_nums)

        # items&users vector
        # self.p_u = np.zeros(self.user_nums, self.k, dtype=np.float)
        self.p_u = np.random.randn(self.user_nums, self.k)
        # self.q_i = np.zeros(self.item_nums, self.k, dtype=np.float)
        self.q_i = np.random.randn(self.item_nums, self.k)

        # implicit preference
        # 形状要和p_u的维度
        # self.y = np.zeros(self.item_nums, self.k, dtype=np.float)
        self.y = np.random.randn(self.item_nums, self.k)

        # 构建binary matrix，这里用字典代替
        self.user_item_mat = {}
        for i in range(len(self.data)):
            user, item, rating = self.data[i, :]
            if user not in self.user_item_mat.keys():
                self.user_item_mat[user] = [item]
            else:
                self.user_item_mat[user].append(item)

    # 采用论文中公式(14)给出的参数
    def train(self, epochs=30, gamma=0.002, Lambda=0.04):
        print('start training!')
        for epoch in range(epochs):

            # 将0~len(self.Data)-1的数据打乱，并返回一个列表
            index_list = np.random.permutation(len(self.data))
            loss = 0

            # 遍历整个training Data
            for i in range(len(self.data)):
                index = index_list[i]
                user, item, rating = self.data[index, :]

                # 论文中N(u)的含义为用户u所交互过的所有物品
                N_u = self.user_item_mat[user]

                # 式子中最后那部分的求和
                y_j = np.sum(self.y[N_u], axis=0)
                # 标准化
                y_j = y_j / np.sqrt(len(N_u))

                # 预测
                pred = self.mu + self.b_u[user] + self.b_i[item] + np.dot(self.q_i[item], self.p_u[user] + y_j)

                e_ui = rating - pred
                loss += e_ui ** 2

                # sgd，虽然loss是取平方，但实际上会乘上1/2，这样在求梯度的时候式子会更好看一点

                # 求梯度，并且往梯度的反方向前进
                self.b_u[user] += gamma * (e_ui - Lambda * self.b_u[user])
                self.b_i[item] += gamma * (e_ui - Lambda * self.b_i[item])
                self.p_u[user] += gamma * (e_ui * self.q_i[item] - Lambda * self.p_u[user])
                self.q_i[item] += gamma * (e_ui * (self.p_u[user] + y_j) - Lambda * self.q_i[item])
                #
                for j in N_u:
                    self.y[j] += gamma * (e_ui * self.q_i[item] / np.sqrt(len(N_u)) - Lambda * self.y[j])

            print(f'epoch: {epoch + 1}, train_loss(rmse): {np.sqrt(loss / len(self.data))}')

    def test(self, data):
        data = np.array(data)

        loss = 0
        for i in range(len(data)):
            user, item, rating = data[i, :]

            N_u = self.user_item_mat[user]

            y_j = np.sum(self.y[N_u], axis=0)
            y_j = y_j / np.sqrt(len(N_u))

            # predict
            pred = self.mu + self.b_u[user] + self.b_i[item] + np.dot(self.q_i[item], self.p_u[user] + y_j)

            e_ui = rating - pred
            loss += e_ui ** 2

        print(f'test_loss(rmse): {np.sqrt(loss / len(data))}')
