import numpy as np
import torch

data = np.array([1, 2, 3])
# Tensor是类构造函数，tensor是工厂函数
# tensor还能指定数据类型，允许接受的参数更多
t = torch.tensor(data, dtype=torch.float64)
print(t.dtype)

# Tensor和tensor在改变原数据后，张量不会发生改变
# 而from_numpy和as_tensor的张量则会跟随原数据改变

# 结论：tensor最常用，如果必须要优化内存也是用as_tensor(from_numpy只能接受numpy数组)
