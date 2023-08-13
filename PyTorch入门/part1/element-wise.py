import torch

# element-wise的操作保证有对应数量的轴，且对应轴的长度相等
# 如果使用两个形状不同的数据进行element-wise操作，则会在幕后自动进行广播机制操作，即复制一个或多个轴使二者形状形同
t1 = torch.tensor([
    [1, 1],
    [1, 1]
], dtype=torch.float32)

t2 = torch.tensor([2, 4], dtype=torch.float32)

print(t1 + t2)
