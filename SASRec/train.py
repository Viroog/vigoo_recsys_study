import argparse
import numpy as np
import torch

from data import Data, WarpSampler
from model import SASRec, SASRec2
from evaluation import Evaluation
import torch.optim as optim
import torch.nn as nn

# 不加这个无法进行多线程处理，会报错
if __name__ == '__main__':

    # 整合参数，比较标准
    parser = argparse.ArgumentParser()
    # parser.add_argument('--path', default='./Data/ml-1m.txt')
    parser.add_argument('--path', default='./my_processed_data/ml-1m.txt')
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--lr', default=0.001)
    parser.add_argument('--epochs', default=200)
    parser.add_argument('--n', default=200)
    parser.add_argument('--d', default=50)
    parser.add_argument('--block_nums', default=2)
    parser.add_argument('--dropout_rate', default=0.2)
    parser.add_argument('--head_nums', default=1)

    args = parser.parse_args()

    data = Data(path=args.path)
    data.data_partition()
    user_train, user_valid, user_test, user_nums, item_nums = data.user_train, data.user_valid, data.user_test, data.user_nums, data.item_nums

    sampler = WarpSampler(user_train, user_nums, item_nums, args.batch_size, args.n, 3)
    batch_nums = len(user_train) // args.batch_size

    # sasrec = SASRec(args.n, item_nums, args.d, args.dropout_rate, args.block_nums, args.head_nums).cuda()
    sasrec = SASRec2(args.n, item_nums, args.d, args.dropout_rate, args.block_nums, args.head_nums, use_conv=False).cuda()

    # 这个初始化参数很重要，指标提升了很多
    for name, param in sasrec.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass  # just ignore those failed init layers

    optimizer = optim.Adam(sasrec.parameters(), lr=args.lr, betas=(0.9, 0.98))
    # optimizer = optim.AdamW(sasrec.parameters(), lr=args.lr)

    sasrec.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in range(batch_nums):
            user, seq, pos, neg = sampler.next_batch()
            # (batch_size, n)
            user, seq, pos, neg = np.array(user), np.array(seq), np.array(pos), np.array(neg)

            pos_pred, neg_pred = sasrec(seq, pos, neg)
            pos_label = torch.ones_like(pos_pred).cuda()
            neg_label = torch.zeros_like(neg_pred).cuda()

            # 在计算loss的时候，论文中忽略了o_t=<pad>
            # 这里是pos!=0的地方，输入的最后一个padding的输出是有确切物品输出的
            idxs = np.where(pos != 0)

            criterion = nn.BCEWithLogitsLoss()

            # 正样本的loss
            loss = criterion(pos_pred[idxs], pos_label[idxs])
            # 负样本的loss
            loss += criterion(neg_pred[idxs], neg_label[idxs])

            total_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"epoch: {epoch}, loss: {total_loss / args.batch_size}")

        # 20个epoch在验证集和测试集上验证模型性能
        if (epoch + 1) % 20 == 0:
            sasrec.eval()
            evaluation = Evaluation(sasrec, data, args, 10)
            valid_ndcg, valid_hit = evaluation.get_validation_metric()
            test_ndcg, test_hit = evaluation.get_test_metric()
            print(f"Validation, NDCG@10: {valid_ndcg}, HIT@10: {valid_hit}")
            print(f"Test, NDCG@10: {test_ndcg}, HIT@10: {test_hit}")
            sasrec.train()

    sampler.close()
