import torch

t1 = torch.tensor([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
])

t2 = torch.tensor([
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1]
])

t3 = torch.tensor([
    [2, 2, 2, 2],
    [2, 2, 2, 2],
    [2, 2, 2, 2],
    [2, 2, 2, 2]
])

# 叠加
# 一个张量就是一个batch
t = torch.stack((t1, t2, t3))
print(t.shape)

# 像图像一样多增加一个颜色通道
t = t.reshape(3, 1, 4, 4)

# 对图像进行预测时，我们需要的是将每个图像扁平化，而保持批次的维度不变，使用flatten并指定开始压缩的维度
print(t.flatten(start_dim=1).shape)