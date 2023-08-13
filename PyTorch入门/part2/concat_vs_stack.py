import torch

# concat是在已有的轴上进行拼接
# stack是在新的轴上进行拼接
t1 = torch.tensor([1, 1, 1])
t2 = torch.tensor([2, 2, 2])
t3 = torch.tensor([3, 3, 3])

t = torch.cat((t1, t2, t3), dim=0)
print(t)

# 
t = torch.stack((t1, t2, t3), dim=1)
print(t)