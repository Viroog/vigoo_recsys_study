from data import Data
from model import ItemCF, UserCF
from evaluation import Evaluation

train_data, test_data = Data("../data/ml-100k/").get_data()
print("load Data finished...")
user_nums = 943
item_nums = 1682
K = 50
icf = ItemCF(user_nums, item_nums, K)
print("start training ICF...")
icf.train(train_data)
ranking, ground_truth = icf.test(test_data)

# evaluation
k = 5
evaluation = Evaluation(ranking, ground_truth, k)
print("Item Based OCCF:")
print(f"Precision@{k}: {evaluation.precision()}")
print(f"Recall@{k}: {evaluation.recall()}")


ucf = UserCF(user_nums, item_nums, K)
print("start training UCF...")
ucf.train(train_data)
ranking, ground_truth = ucf.test(test_data)

evaluation = Evaluation(ranking, ground_truth, k)
print("Uer Base OCCF:")
print(f"Precision@{k}: {evaluation.precision()}")
print(f"Recall@{k}: {evaluation.recall()}")