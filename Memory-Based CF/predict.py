from data import Data
from model import UCF, ICF, HCF

train_data, test_data = Data().get_data()

# User-Based CF
ucf = UCF(data=train_data)
ucf.train()
# 返回参数是numpy数组
pred_ucf = ucf.test(data=test_data)

#
# # Item-Based CF
# icf = ICF(data=train_data)
# icf.train()
# # 返回参数是numpy数组
# pred_icf = icf.test(data=test_data)
#
# # Hybrid CF
# hcf = HCF(pred_ucf=pred_ucf, pred_icf=pred_icf, Lambda=0.5)
# hcf.test(data=test_data)