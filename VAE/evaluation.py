import numpy as np


class Evaluation:
    def __init__(self, y, y_hat, k):
        self.y = y
        self.y_hat = y_hat
        self.k = k

    def precision(self):

        total_precision = 0
        cnt = 0

        for i in range(len(self.y)):
            topk_items = np.argsort(self.y_hat[i, :])[-self.k:]

            Iu = list(np.where(self.y[i, :] != 0.0)[0])

            if len(Iu) > 0:
                cnt += 1
                total_precision += len(set(topk_items).intersection(set(Iu))) / self.k

        return total_precision / cnt

    def recall(self):
        total_recall = 0
        cnt = 0

        for i in range(len(self.y)):
            topk_items = np.argsort(self.y_hat[i, :])[-self.k:]

            Iu = list(np.where(self.y[i, :] != 0.0)[0])

            if len(Iu) > 0:
                cnt += 1
                total_recall += len(set(topk_items).intersection(set(Iu))) / len(Iu)

        return total_recall / cnt

    def ndcg(self):
        pass

    def mrr(self):
        pass

