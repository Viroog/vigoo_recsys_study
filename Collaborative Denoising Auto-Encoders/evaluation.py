import numpy as np


class Evaluation:
    def __init__(self, y, y_hat, topk):
        self.y = y
        self.y_hat = y_hat
        self.topk = topk

    def precision(self):

        total_precision = 0
        cnt = 0

        for i in range(self.y.shape[0]):
            # 获得值最高的k个
            topk_items = np.argsort(self.y_hat[i, :])[-self.topk:]

            # 取出测试集中用户买过的物品
            Iu = list(np.where(self.y[i, :] != 0.0)[0])

            if len(Iu) > 0:
                cnt += 1
                total_precision += len(set(Iu).intersection(set(topk_items))) / self.topk

        return total_precision / cnt

    def recall(self):
        total_recall = 0
        cnt = 0

        for i in range(self.y.shape[0]):
            # 获得值最高的5个
            topk_items = np.argsort(self.y_hat[i, :])[-self.topk:]

            # 取出测试集中用户买过的物品
            Iu = list(np.where(self.y[i, :] != 0.0)[0])

            if len(Iu) > 0:
                cnt += 1
                total_recall += len(set(Iu).intersection(set(topk_items))) / len(Iu)

        return total_recall / cnt

