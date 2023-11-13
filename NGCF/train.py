import argparse
import os.path

import torch

from data import Data
from model import NGCF
import torch.optim as optim

argparse = argparse.ArgumentParser()

argparse.add_argument('--epochs', type=int, default=400)
argparse.add_argument('--batch_size', type=int, default=1024)
argparse.add_argument('--lr', type=float, default=1e-4)
argparse.add_argument('--l2_reg', type=float, default=1e-5)
argparse.add_argument('--embed_size', type=int, default=64)
argparse.add_argument('--layer_size', type=str, default='[64, 64, 64]', help='output size of every layer')

argparse.add_argument('--node_dropout', type=int, default=0, help='1 means apply node dropout and 0 means do not')
argparse.add_argument('--node_dropout_rate', type=float, default=0.2)
argparse.add_argument('--message_dropout_rate', type=float, default=0.1)

argparse.add_argument('--device', type=str, default='cuda:0')
argparse.add_argument('--dataset', type=str, default='gowalla', choices=['amazon-book', 'gowalla'])
argparse.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])
argparse.add_argument('--model_path', type=str, default='ngcf.pth')
argparse.add_argument('--max_endure', type=int, default=50)

args = argparse.parse_args()

# use to implement early stop
best_ndcg = float('-inf')
cnt_endure = 0
# training
if args.mode == 'train' or (args.mode == 'eval' and os.path.exists(args.model_path) is False):

    data = Data(args.dataset, args.batch_size)
    adj_mat, norm_adj_mat, mean_adj_mat = data.get_adj_mat()

    ngcf = NGCF(data.user_nums, data.item_nums, norm_adj_mat, args).to(args.device)
    ngcf.train()

    optimizer = optim.Adam(ngcf.parameters(), lr=args.lr, weight_decay=args.l2_reg)

    for epoch in range(args.epochs):

        batch_steps = data.training_nums // args.batch_size + 1
        total_loss = 0

        for step in range(batch_steps):
            users, pos_items, neg_items = data.sample()

            loss = ngcf(users, pos_items, neg_items)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'epoch: {epoch + 1}, loss={total_loss / batch_steps}')

# eval
else:
    ngcf = torch.load(args.model_path).to(args.device)
    ngcf.eval()
