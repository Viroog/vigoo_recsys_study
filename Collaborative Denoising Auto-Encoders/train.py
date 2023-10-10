import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


from data import Data, DataSet
from evaluation import Evaluation
from model import CDAE

dataloader = Data(path='../data/ml-1m/ratings.dat')
# dataloader = Data(path='../data/ml-20m/ratings.csv')
train_data, test_data, user_nums, item_nums = dataloader.train_data, dataloader.test_data, dataloader.user_nums, dataloader.item_nums
hidden_size, corrupted_ration = 50, 0.5
cdae = CDAE(user_nums, item_nums, hidden_size, corrupted_ration)

batch_size, lr, epochs = 128, 0.001, 300
# negative sampling
ns = 3
optimizer = optim.Adam(cdae.parameters(), lr=lr)

# 构建一个pytorch的dataloader类 主要用到torch.utils.data.DataLoader和torch.utils.data.DataSet
# Dataloader接收DataSet作为输入
train_dataset, test_dataset = DataSet(train_data), DataSet(test_data)
train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size)

# 开启训练模式，在这里指打开dropout层
cdae.train()
for epoch in range(epochs):
    total_loss = 0
    # 每一个epoch训练一遍数据，一遍中分成多个batch
    for batch in train_loader:
        user_ids, input_mat = batch

        user_ids = user_ids.cuda()
        input_mat = input_mat.float().cuda()

        # 这里能直接这样调用是因为继承了nn.Module类，里面实现了特殊的forward调用方法
        preds = cdae.forward(user_ids, input_mat)

        # # 或许计算loss的时候有问题，我这里好像用了全部的负样本，应该只用某些负样本，否则会导致样本数量偏差过大，导致训练效果不是很好(过拟合)
        # # 做法：将预测值的某些原本为0的数据删除，只采样部分
        # # 负采样物品及其对应的用户
        # negative_items, users = [], []
        # for user in range(input_mat.shape[0]):
        #     # 每个用户的负样本数量为正样本的数量的ns倍
        #     # 可能不足五倍，就只能取全部
        #     num_ns_per_user = min(int(ns * input_mat[user].sum()), int(item_nums - input_mat[user].sum()))
        #     # 找出负样本的索引，即物品编号
        #     zero_idx = torch.where(input_mat[user] == 0.)[0]
        #     # 采样
        #     random_idx = torch.randperm(len(zero_idx))[:num_ns_per_user]
        #     ns_indices = zero_idx[random_idx]
        #     # 记录
        #     negative_items.extend(ns_indices.cpu().numpy())
        #     users.extend([user] * num_ns_per_user)
        #
        # mask = input_mat.clone()
        # # mask掩码：里面为1的值是原本已有的数据以及新采样的负样本数据
        # mask[users, negative_items] = 1
        #
        # # mask值为1的地方保留原来的值，为保留的值全部置0
        # masked_input_mat = input_mat[mask > 0.]
        # masked_preds = preds[mask > 0.]

        # 计算损失
        # BCELoss的输出是未经过sigmoid的，BCEWithLogitsLoss的输出是经过sigmoid的
        loss_function = nn.BCELoss()

        # loss = loss_function(input=masked_preds, target=masked_input_mat)
        loss = loss_function(input=preds, target=input_mat)
        # 清空梯度，防止梯度的累积
        optimizer.zero_grad()
        loss.backward()
        # 更新梯度
        optimizer.step()

        total_loss += loss

    print(f"epoch: {epoch + 1}, loss: {total_loss / user_nums}")

# model_path = 'cdae.pth'
# torch.save(cdae, model_path)

# cdae = torch.load(model_path)
# 关闭dropout层，之前会不会是这里出了问题
cdae.eval()
preds = cdae.predict(loader=train_loader)

topk = 5
evaluation = Evaluation(y=test_data, y_hat=preds, topk=topk)
print(f"Pre@{topk}: {evaluation.precision()}")
print(f"Rec@{topk}: {evaluation.recall()}")
