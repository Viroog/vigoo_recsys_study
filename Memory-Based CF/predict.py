import numpy as np

from data import Data
from model import UCF, ICF, HCF

np.set_printoptions(precision=4)

train_data, test_data = Data().get_data()

# User-Based CF
ucf = UCF(data=train_data)
ucf.train()
# 返回参数是numpy数组
pred_ucf = ucf.test(data=test_data)
np.savetxt('pred_ucf_without_abs.txt', pred_ucf)

# Item-Based CF
icf = ICF(data=train_data)
icf.train()
# 返回参数是numpy数组
pred_icf = icf.test(data=test_data)
np.savetxt('pred_icf.txt', pred_icf)

# Hybrid CF
hcf = HCF(Lambda=0.5)
hcf.test(data=test_data)
