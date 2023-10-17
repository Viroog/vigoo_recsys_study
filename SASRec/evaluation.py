import numpy as np


# 验证集
class Evaluation:
    def __init__(self, model, dataset, args, k):
        self.model = model
        self.user_train, self.user_valid, self.user_test, self.user_nums, self.item_nums = dataset.user_train, dataset.user_valid, dataset.user_test, dataset.user_nums, dataset.item_nums
        self.args = args
        self.k = k

    def get_validation_metric(self):

        total_ndcg = 0
        total_hit = 0
        cnt = 0

        users = list(self.user_train.keys())
        for user in users:
            if len(self.user_train[user]) < 1 or len(self.user_valid[user]) < 1:
                continue

            cnt += 1

            seq = np.zeros([self.args.n], dtype=np.int32)
            idx = self.args.n - 1

            # 获得用户的交互序列
            for item in reversed(self.user_train[user]):
                seq[idx] = item
                idx -= 1

                if idx == -1:
                    break

            Iu = set(self.user_train[user])
            Iu.add(0)
            item_idxs = [self.user_valid[user][0]]
            # 除了验证集，再随机抽取100个negative items
            for _ in range(100):
                item = np.random.randint(1, self.item_nums + 1)
                # 要采集的是negative item，不能出现在Iu中
                while item in Iu:
                    item = np.random.randint(1, self.item_nums + 1)
                item_idxs.append(item)

            # [seq]是为了升维，使得和训练的时候输入形状一致，第一维为batch_size，即用户个数
            # 加了个负号
            preds = -self.model.predict(*[np.array(elem) for elem in [[seq], item_idxs]])
            # 最终变成一维，即101个物品的预测评分
            preds = preds[0]

            # 调用两次argsort的得到的最终结果原数组中第一个元素(即验证集中的物品)在降序排名中的位置
            rank = preds.argsort().argsort()[0].item()

            # 如果验证集出现在前k个就算成功
            if rank < self.k:
                total_ndcg += 1 / np.log2(rank + 2)
                total_hit += 1

        return total_ndcg / cnt, total_hit / cnt

    def get_test_metric(self):

        total_ndcg = 0
        total_hit = 0
        cnt = 0

        users = list(self.user_train.keys())
        for user in users:
            if len(self.user_train[user]) < 1 or len(self.user_train[user]) < 1:
                continue

            cnt += 1

            seq = np.zeros([self.args.n], dtype=np.int32)
            idx = self.args.n - 1
            # 测试的时候，把验证集也放进去
            seq[idx] = self.user_valid[user][0]
            idx -= 1

            # 获得用户的交互序列
            for item in reversed(self.user_train[user]):
                seq[idx] = item
                idx -= 1
                if idx == -1:
                    break

            Iu = set(self.user_train[user])
            Iu.add(self.user_valid[user][0])
            Iu.add(0)
            item_idxs = [self.user_test[user][0]]
            # 除了验证集，再随机抽取100个negative items
            for _ in range(100):
                item = np.random.randint(1, self.item_nums + 1)
                # 要采集的是negative item，不能出现在Iu中
                while item in Iu:
                    item = np.random.randint(1, self.item_nums + 1)
                item_idxs.append(item)

            # [seq]是为了升维，使得和训练的时候输入形状一致，第一维为batch_size，即用户个数
            # 加了个负号
            preds = -self.model.predict(*[np.array(elem) for elem in [[seq], item_idxs]])
            # 最终变成一维，即101个物品的预测评分
            preds = preds[0]

            # 调用两次argsort的得到的最终结果原数组中第一个元素(即验证集中的物品)在降序排名中的位置
            rank = preds.argsort().argsort()[0].item()

            # 如果验证集出现在前k个就算成功
            if rank < self.k:
                total_ndcg += 1 / np.log2(rank + 2)
                total_hit += 1

        return total_ndcg / cnt, total_hit / cnt
