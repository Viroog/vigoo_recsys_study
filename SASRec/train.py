import argparse
import numpy as np

from data import Data, Sampler
from model import SASRec

# 整合参数，比较标准
parser = argparse.ArgumentParser()
parser.add_argument('--path', default='./data/ml-1m.txt')
parser.add_argument('--batch_size', default=128)
parser.add_argument('--lr', default=0.001)
parser.add_argument('--optimizer', default='Adam')
parser.add_argument('--epochs', default=200)
parser.add_argument('--n', default=200)
parser.add_argument('--d', default=50)
parser.add_argument('--block_nums', default=2)
parser.add_argument('--dropout_rate', default=0.2)

args = parser.parse_args()

data = Data(path=args.path)
data.data_partition()
user_train, user_valid, user_test, user_nums, item_nums = data.user_train, data.user_valid, data.user_test, data.user_nums, data.item_nums

sampler = Sampler(user_train, user_nums, item_nums, args.batch_size, args.n)
batch_nums = len(user_train) // args.batch_size

sasrec = SASRec(args.n, item_nums, args.d, args.dropout_rate)

sasrec.train()
for epoch in range(args.epochs):
    for batch in range(batch_nums):
        user, seq, pos, neg = sampler.get_one_batch()
        # (batch_size, n)
        user, seq, pos, neg = np.array(user), np.array(seq), np.array(pos), np.array(neg)

        sasrec.forward()