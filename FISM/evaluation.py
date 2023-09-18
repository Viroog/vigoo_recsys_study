class Evaluation:
    def __init__(self, k, ranking_u_i_dict, test_u_i_dict):
        self.k = k
        self.ranking_u_i_dict = ranking_u_i_dict
        self.test_u_i_dict = test_u_i_dict
        self.test_u_nums = len(test_u_i_dict)

    def precision(self):

        total_precision = 0

        for k, v1 in self.test_u_i_dict.items():
            v2 = self.ranking_u_i_dict[k]

            total_precision += len(set(v1).intersection(set(v2[:self.k]))) / self.k

        return total_precision / self.test_u_nums

    def recall(self):
        total_recall = 0

        for k, v1 in self.test_u_i_dict.items():
            v2 = self.ranking_u_i_dict[k]

            total_recall += len(set(v1).intersection(set(v2[:self.k]))) / len(v1)

        return total_recall / self.test_u_nums
