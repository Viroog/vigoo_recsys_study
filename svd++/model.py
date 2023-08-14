import numpy as np

# 随机数种子
# np.random.seed(42)


class SVDpp:
    def __init__(self, data, k=20):
        # training data
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
        # 形状要和p_u一致
        # self.y = np.zeros(self.user_nums, self.k, dtype=np.float)
        self.y = np.random.randn(self.user_nums, self.k)

        # 构建binary matrix，这里用字典代替
        self.user_item_mat = {}
        for i in range(len(self.data)):
            user, item, rating = self.data[i, :]
            if user not in self.user_item_mat.keys():
                self.user_item_mat[user] = [item]
            else:
                self.user_item_mat[user].append(item)

    # 采用论文中公式(14)给出的参数
    def train(self, epoch=30, step=0.002, Lambda=0.04):
        pass
