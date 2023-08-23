from data import Data
from model import PopRank
from ranking_evaluation import Evaluation

train_data, test_data = Data().get_data()
model = PopRank(train_data)
model.train()

k = 5
pred_rec, real_buy = model.test(test_data, k)

evaluation = Evaluation(pred_rec, real_buy, k)
# 评估指标
print(f"Pre@{k}:", evaluation.precision())
print(f"Rec@{k}:", evaluation.recall())
print(f"F1@{k}:", evaluation.F1())
