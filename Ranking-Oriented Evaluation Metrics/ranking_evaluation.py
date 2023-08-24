import numpy as np


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
        for i, (k, v) in enumerate(self.pred_rec.items()):

            pred, real = v, self.real_buy[k]

            precision_u = len(set(pred[:self.k]).intersection(set(real))) / self.k
            total_precision_u += precision_u

        return total_precision_u / self.u_test_nums

    def recall(self):
        total_recall_u = 0

        # k是用户，v是推荐物品序列
        for (k, v) in self.pred_rec.items():
            pred, real = v, self.real_buy[k]

            recall_u = len(set(pred[:self.k]).intersection(set(real))) / len(real)
            total_recall_u += recall_u

        return total_recall_u / self.u_test_nums

    def F1(self):
        total_f1_u = 0

        for (k, v) in self.pred_rec.items():
            pred, real = v, self.real_buy[k]

            precision_u = len(set(pred[:self.k]).intersection(set(real))) / self.k
            recall_u = len(set(pred[:self.k]).intersection(set(real))) / len(real)

            if precision_u + recall_u != 0:
                f1_u = 2 * ((precision_u * recall_u) / (precision_u + recall_u))
            else:
                f1_u = 0

            total_f1_u += f1_u

        return total_f1_u / self.u_test_nums

    def DCG(self, user):
        total_dcg = 0
        total_idcg = 0

        rec_list = self.pred_rec[user]
        item_list = self.real_buy[user]

        # 理想状态为：测试集与推荐集的交集，而不是按照预测的分数来
        idea_len = len(set(rec_list).intersection(set(item_list)))

        # l从1开始
        for l in range(1, self.k + 1):
            # 列表从0开始
            if rec_list[l - 1] in item_list:
                dcg_u = 1 / np.log(l + 1)
            else:
                dcg_u = 0

            if l <= idea_len:
                idcg_u = 1 / np.log(l + 1)
            else:
                idcg_u = 0

            total_dcg += dcg_u
            total_idcg += idcg_u

        return total_dcg, total_idcg

    def NDCG(self):
        total_ndcg = 0

        for k, v in self.pred_rec.items():
            dcg_u, idcg_u = self.DCG(k)
            total_ndcg += dcg_u / idcg_u

        return total_ndcg / self.u_test_nums

    # 至少出现一个
    def one_call(self):
        total_one_call = 0

        for k, v in self.pred_rec.items():
            pred, real = v, self.real_buy[k]

            if len(set(pred[:self.k]).intersection(set(real))) > 0:
                total_one_call += 1

        return total_one_call / self.u_test_nums

    # rank: 推荐列表中第一个在ground_truth结果中的item所在的排列位置
    def MRR(self):
        total_rr = 0

        for k, v in self.pred_rec.items():
            pred, real = v, self.real_buy[k]

            for i, item in enumerate(pred):
                if item in real:
                    total_rr += 1 / (i+1)
                    break

        return total_rr / self.u_test_nums

    def MAP(self):
        total_ap = 0

        for k, v in self.pred_rec.items():
            pred, real = v, self.real_buy[k]

            ap_u = 0

            for i, item in enumerate(pred):
                if item in real:
                    if i == 0:
                        tmp = 1 / (i + 1)
                    else:
                        tmp = (len(set(pred[:i]).intersection(set(real))) + 1) / (i + 1)

                    ap_u += tmp

            total_ap += ap_u / len(real)

        return total_ap / self.u_test_nums

    def ARP(self, total_item, u_i_dict):
        total_rp = 0

        for k, v in self.pred_rec.items():
            pred, real = v, self.real_buy[k]

            tmp = 0

            for i, item in enumerate(pred):
                if item in real:
                    # I:the whole item set
                    # Iu: preferred items by user u in training data
                    # 分母: I - Iu
                    tmp += (i + 1) / (total_item - len(u_i_dict[k]))

            rp_u = tmp / len(real)

            total_rp += rp_u

        return total_rp / self.u_test_nums

    # 正样本的预测结果大于负样本的预测结果的概率，反映的是模型对样本的排序能力
    # 这里的正样本的是测试集中的(u,i)对，负样本是从未出现在训练集和测试集中的(u,j)对
    def AUC(self, train_data, test_data, total_item, b_i):

        R, R_te = {}, {}

        for i in range(len(train_data)):
            user, item, rating = train_data[i, :]

            if user not in R.keys():
                R[user] = [item]
            else:
                R[user].append(item)

        for i in range(len(test_data)):
            user, item, rating = test_data[i, :]

            if user not in R_te.keys():
                R_te[user] = [item]
            else:
                R_te[user].append(item)

        total_AUC = 0
        # list1中的为正样本，list2中的为负样本
        for u, list1 in R_te.items():
            if u in R.keys():
                list2 = R[u]

            list3 = list(set(total_item) - set(list1) - set(list2))

            AUC_u = 0
            cnt = 0

            for i in list1:
                for j in list3:
                    if i in b_i.keys() and j in b_i.keys():
                        if b_i[i] > b_i[j]:
                            AUC_u += 1
                    else:
                        cnt += 1

            total_AUC += AUC_u / (len(list1) * len(list3) - cnt)

        return total_AUC / self.u_test_nums
