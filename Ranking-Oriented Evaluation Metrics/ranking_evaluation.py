# 使用poprank指标来充当预测分数

class Evaluation:
    def __init__(self, pred_rec, real_buy, k):
        self.pred_rec = pred_rec
        self.real_buy = real_buy
        self.u_test_nums = len(real_buy)
        self.k = k

    def precision(self):
        total_precision_u = 0

        # k是用户，v是推荐物品序列
        for (k, v) in self.pred_rec.items():
            pred, real = v, self.real_buy[k]

            precision_u = len(set(pred).intersection(set(real))) / self.k
            total_precision_u += precision_u

        return total_precision_u / self.u_test_nums

    def recall(self):
        total_recall_u = 0

        # k是用户，v是推荐物品序列
        for (k, v) in self.pred_rec.items():
            pred, real = v, self.real_buy[k]

            recall_u = len(set(pred).intersection(set(real))) / len(real)
            total_recall_u += recall_u

        return total_recall_u / self.u_test_nums

    def F1(self):
        total_f1_u = 0

        for (k, v) in self.pred_rec.items():
            pred, real = v, self.real_buy[k]

            precision_u = len(set(pred).intersection(set(real))) / self.k
            recall_u = len(set(pred).intersection(set(real))) / len(real)

            if precision_u + recall_u != 0:
                f1_u = 2 * ((precision_u * recall_u) / (precision_u + recall_u))
            else:
                f1_u = 0

            total_f1_u += f1_u

        return total_f1_u / self.u_test_nums

    def NDCG(self):
        pass

    def one_call(self):
        pass

    def MRR(self):
        pass

    def MAP(self):
        pass

    def ARP(self):
        pass

    def AUC(self):
        pass