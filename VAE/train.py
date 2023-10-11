import torch

from data import Data, DataSet
from model import VAE
from evaluation import Evaluation
import torch.utils.data as data
import torch.optim as optim


def my_loss_function(prob, input_mat, mu, logvar, beta=0.2):

    # maximize {log[p(x_u|z_u)] - beta * KL_diversion]}
    # equal to minimize {-log[p(x_u|z_u)] + beta * KL_diversion}

    # i∈Iu, 和input_mat相乘再求和, 再求mean
    multinomial_likelihood = -torch.mean(torch.sum(prob * input_mat, dim=1))

    # 0.5 * (-logvar + mu^2 + var - 1), 然后再对K个维度(hidden_size), 再求mean
    KL_diversion = 0.5 * torch.mean(torch.sum(-logvar + mu.pow(2) + logvar.exp() - 1, dim=1))

    return multinomial_likelihood + beta * KL_diversion


dataloader = Data(path='../data/ml-1m/ratings.dat')
train_data, test_data, user_nums, item_nums = dataloader.train_data, dataloader.test_data, dataloader.user_nums, dataloader.item_nums

hidden_size = 250
drop_ration = 0.5
vae = VAE(user_nums, item_nums, hidden_size, drop_ration)

train_dataset = DataSet(train_data)
test_dataset = DataSet(test_data)

batch_size = 100
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size)

lr = 1e-4
optimizer = optim.Adam(vae.parameters(), lr=lr)

# epochs = 1000
# vae.train()
# for epoch in range(epochs):
#     total_loss = 0
#     for batch in train_dataloader:
#         user_ids, input_mat = batch
#
#         input_mat = input_mat.float().cuda()
#
#         # prob: (batch_size, item_nums)
#         # mu: (batch_size, hidden_size)
#         # logvar: (batch_size, hidden_size)
#         prob, mu, logvar = vae.forward(input_mat)
#
#         loss = my_loss_function(prob, input_mat, mu, logvar)
#
#         optimizer.zero_grad()
#         loss.backward()
#
#         optimizer.step()
#
#         total_loss += loss
#
#     print(f"epoch: {epoch + 1}, loss: {total_loss / len(train_dataloader)}")

# torch.save(vae, 'vae.pth')

vae = torch.load('vae.pth')

vae.eval()
preds = vae.predict(train_dataloader)

k = 5
evaluation = Evaluation(y=test_data, y_hat=preds, k=k)
print(f"Pre@{k}: {evaluation.precision()}")
print(f"Rec@{k}: {evaluation.recall()}")
print(f"NDCG@{k}: {evaluation.ndcg()}")
print(f"MRR@{k}: {evaluation.mrr()}")