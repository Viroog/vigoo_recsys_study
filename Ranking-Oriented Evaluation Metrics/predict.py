from data import Data
from model import PopRank
from ranking_evaluation import Evaluation

train_data, test_data, total_item = Data().get_data()
model = PopRank(train_data)
u_i_dict, b_i = model.train(return_val=1)

k = 5
pred_rec, real_buy = model.test(test_data)

evaluation = Evaluation(pred_rec, real_buy, k)
# 评估指标
print(f"Pre@{k}:", evaluation.precision())
print(f"Rec@{k}:", evaluation.recall())
print(f"F1@{k}:", evaluation.F1())
print(f"NDCG@{k}:", evaluation.NDCG())
print(f"1-call@{k}:", evaluation.one_call())
print("MRR:", evaluation.MRR())
print("MAP:", evaluation.MAP())
print("ARP:", evaluation.ARP(len(total_item), u_i_dict))
print("AUC:", evaluation.AUC(train_data, test_data, total_item, b_i))
