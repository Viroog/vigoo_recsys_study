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

        # items&users vector
        self.p_u = np.random.randn(self.user_nums, self.k)
        self.q_i = np.random.randn(self.item_nums, self.k)

    # r_ui = q_i * p_u
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

                # 预测
                pred = np.dot(self.q_i[item], self.p_u[user])

                e_ui = rating - pred
                loss += e_ui ** 2

                # sgd，虽然loss是取平方，但实际上会乘上1/2，这样在求梯度的时候式子会更好看一点
                # 求梯度，并且往梯度的反方向前进
                self.p_u[user] += gamma * (e_ui * self.q_i[item] - Lambda * self.p_u[user])
                self.q_i[item] += gamma * (e_ui * self.p_u[user] - Lambda * self.q_i[item])

            # 第一个epoch要以w模式打开，没有自动创建
            if epoch == 0:
                with open('./log.txt', 'w') as f:
                    f.write(f'epoch: {epoch + 1}, train_loss(rmse): {np.sqrt(loss / len(self.data))}\n')
                f.close()
            # 其他epoch，则以a模式进行追加
            else:
                with open('./log.txt', 'a') as f:
                    f.write(f'epoch: {epoch + 1}, train_loss(rmse): {np.sqrt(loss / len(self.data))}\n')
                f.close()

            print(f'epoch: {epoch + 1}, train_loss(rmse): {np.sqrt(loss / len(self.data))}')

    def test(self, data):
        data = np.array(data)

        loss = 0
        for i in range(len(data)):
            user, item, rating = data[i, :]

            # predict
            pred = np.dot(self.q_i[item], self.p_u[user])

            e_ui = rating - pred
            loss += e_ui ** 2

        with open('./log.txt', 'a') as f:
            f.write(f'test_loss(rmse): {np.sqrt(loss / len(data))}\n')
        f.close()

        print(f'test_loss(rmse): {np.sqrt(loss / len(data))}')
