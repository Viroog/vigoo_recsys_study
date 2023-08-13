import torch

# 该对象在默认情况下是在CPU上进行的
t = torch.tensor([1, 2, 3])
print(t)

# 调用cuda函数依然会返回一个张量，不同的是该张量位于GPU上
t = t.cuda()
print(t)

t = torch.Tensor()
print(t.layout)

# Tensor Attribute
print(t.dtype)
print(t.device)
print(t.layout)

# 张量的计算要在同一个设备上
# t1 = torch.tensor([1, 2, 3])
# t2 = t1.cuda()
# RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
# print(t2 + t1)

# 单位矩阵, 指定行数，返回方阵
# 对角线上为1，其余全为0
t = torch.eye(3)
print(t)

# 零矩阵，指定行和列
t = torch.zeros(2, 3)
print(t)

# 一矩阵，指定行和列
t = torch.ones(2, 3)
print(t)

# 随机矩阵
t = torch.rand(2, 3)
print(t)
