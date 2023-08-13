import torch

# reduction就是减少张量中包含的元素数量的操作
t = torch.tensor([
    [0, 1, 0],
    [2, 0, 2],
    [0, 3, 0]
], dtype=torch.float32)

# 求和操作是一个reduction op
# 不指定维度则对所有元素进行操作
print(t.sum(), t.sum().numel())

# 也可以指定特定的维度
# t.sum(dim=0) equal t[0] + t[1] + t[2] 本质还是element-wise
print(t.sum(dim=0))

# argmax函数没有指定轴时，返回的结果是flatten后计算得到的结果
