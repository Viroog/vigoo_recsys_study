class Evaluation:
    def __init__(self, ranking, ground_truth, k):
        self.ranking = ranking
        self.ground_truth = ground_truth
        self.test_user_nums = len(self.ground_truth)
        self.k = k

    def precision(self):

        total_precision = 0

        for k, v in self.ranking.items():
            ranking, ground_truth = v, self.ground_truth[k]

            precision_u = len(set(ranking[:self.k]).intersection(set(ground_truth))) / self.k

            total_precision += precision_u

        return total_precision / self.test_user_nums

    def recall(self):

        total_recall = 0

        for k, v in self.ranking.items():
            ranking, ground_truth = v, self.ground_truth[k]

            precision_u = len(set(ranking[:self.k]).intersection(set(ground_truth))) / len(ground_truth)

            total_recall += precision_u

        return total_recall / self.test_user_nums
