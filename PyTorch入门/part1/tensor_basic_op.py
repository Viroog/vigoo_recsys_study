import torch

t = torch.tensor([
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [3, 3, 3, 3],
], dtype=torch.float32)

# equal t.size()
print(t.shape)

# number of element
print(t.numel())

# squeeze 移除所有长度为1的轴
print(t.reshape(1, 12).squeeze())
print(t.reshape(1, 12).squeeze().shape)

# unsqueeze 增加一个长度为1的维度
print(t.reshape(1, 12).unsqueeze(dim=0))
print(t.reshape(1, 12).unsqueeze(dim=0).shape)