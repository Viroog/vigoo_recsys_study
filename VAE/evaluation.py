import numpy as np


class Evaluation:
    def __init__(self, y, y_hat, k):
        self.y = y
        self.y_hat = y_hat
        self.k = k

    def precision(self):

        total_precision = 0
        cnt = 0

        for u in range(len(self.y)):
            topk_items = list(np.argsort(self.y_hat[u, :])[-self.k:])

            Iu = list(np.where(self.y[u, :] != 0.0)[0])

            if len(Iu) > 0:
                cnt += 1
                total_precision += len(set(topk_items).intersection(set(Iu))) / self.k

        return total_precision / cnt

    def recall(self):
        total_recall = 0
        cnt = 0

        for u in range(len(self.y)):
            topk_items = list(np.argsort(self.y_hat[u, :])[-self.k:])

            Iu = list(np.where(self.y[u, :] != 0.0)[0])

            if len(Iu) > 0:
                cnt += 1
                total_recall += len(set(topk_items).intersection(set(Iu))) / len(Iu)

        return total_recall / cnt

    def dcg_u(self, topk_items, Iu):

        dcg_u = 0

        # l从1开始，而我这里从0开始
        for l, item in enumerate(topk_items):
            if item in Iu:
                dcg_u += 1 / np.log(l + 1 + 1)

        return dcg_u

    def idcg_u(self, Iu):

        idcg_u = 0

        for l in range(len(Iu)):
            # 只关注前k个
            if l >= self.k:
                break

            idcg_u += 1 / np.log(l + 1 + 1)

        return idcg_u


    def ndcg(self):

        total_ndcg = 0
        cnt = 0

        for u in range(len(self.y)):

            topk_items = list(np.argsort(self.y_hat[u, :])[-self.k:])
            # 根据预测评分从大到小排序
            topk_items.reverse()

            Iu = list(np.where(self.y[u, :] != 0.)[0])

            if len(Iu) > 0:
                cnt += 1
                total_ndcg += self.dcg_u(topk_items, Iu) / self.idcg_u(Iu)

        return total_ndcg / cnt

    def mrr(self):

        total_rr = 0
        cnt = 0

        for u in range(len(self.y)):
            topk_items = list(np.argsort(self.y_hat[u, :])[-self.k:])
            # 按照预测评分从高到低进行排序
            topk_items.reverse()

            Iu = list(np.where(self.y[u, :] != 0.)[0])

            if len(Iu) > 0:
                cnt += 1
                for pos, item in enumerate(topk_items):
                    if item in Iu:
                        total_rr += 1 / (pos + 1)
                        # 找到第一个就要break
                        break

        return total_rr / cnt
