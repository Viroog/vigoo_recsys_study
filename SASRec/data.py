from collections import defaultdict
from multiprocessing import Process, Queue
import numpy as np


class Data:
    # 这里的文件是与处理过后的，即根据timestamp排好序的
    def __init__(self, path):
        self.path = path
        self.user_nums, self.item_nums = None, None
        self.user_train, self.user_valid, self.user_test = None, None, None

    # 将数据分割为train/validate/test三个部分
    # 1) the most recent action S_u[len(S_u)] for testing
    # 2) the second most recent action S_u[len(S_u) - 1] for validation
    # 3) all remaining actions for training
    def data_partition(self):

        u_i_dict = defaultdict(list)
        max_user_nums, max_item_nums = 0, 0

        with open(self.path, 'r') as f:
            for line in f.readlines():
                user_id, item_id = line.rstrip().split(' ')
                user_id, item_id = int(user_id), int(item_id)

                max_user_nums = max(max_user_nums, user_id)
                max_item_nums = max(max_item_nums, item_id)

                u_i_dict[user_id].append(item_id)

        self.user_nums = max_user_nums
        self.item_nums = max_item_nums

        user_train, user_valid, user_test = {}, {}, {}
        for user_id in u_i_dict:
            user_feedback_num = len(u_i_dict[user_id])

            # 交互数量小于3，无训练集和测试集
            if user_feedback_num < 3:
                user_train[user_id] = u_i_dict[user_id]
                user_valid[user_id] = []
                user_test[user_id] = []
            else:
                # 第一个到倒数第三个
                user_train[user_id] = u_i_dict[user_id][:-2]
                # 倒数第二个
                user_valid[user_id] = [u_i_dict[user_id][-2]]
                # 倒数第一个
                user_test[user_id] = [u_i_dict[user_id][-1]]

        self.user_train = user_train
        self.user_valid = user_valid
        self.user_test = user_test

# #自己写的采样器没问题，但是读取速度太慢了
# class Sampler:
#     # n为序列的长度
#     def __init__(self, user_train, user_nums, item_nums, batch_size, n):
#         self.user_train = user_train
#         self.user_nums = user_nums
#         self.item_nums = item_nums
#         self.batch_size = batch_size
#         self.n = n
#
#         self.total_items = set([item for item in range(1, self.item_nums + 1)])
#
#     # 获取一个batch的数据量
#     def get_one_batch(self):
#         one_batch = []
#
#         for _ in range(self.batch_size):
#             # one_batch: (128,)
#             # 每个元素包含user, seq, pos, neg
#             one_batch.append(self.get_one_user(np.random.randint(2e9)))
#
#         # 重新组织数据结构，先解包，再打包
#         return zip(*one_batch)
#
#     # 随机采样一个用户的序列
#     def get_one_user(self, seed):
#
#         # 随机种子，保证产生的用户不一样
#         np.random.seed(seed)
#         user = np.random.randint(1, self.user_nums + 1)
#
#         # 模型的输入输出长度是相等的，保证训练集长度大于1，这样才能有标签
#         while len(self.user_train[user]) <= 1:
#             user = np.random.randint(user)
#
#         # 初始化为0，代表padding
#         # seq为用于训练的序列
#         seq = np.zeros(self.n, dtype=int)
#         # pos和neg为正负标签
#         pos = np.zeros(self.n, dtype=int)
#         neg = np.zeros(self.n, dtype=int)
#
#         # nxt为当前输入对应的标签
#         nxt = self.user_train[user][-1]
#         idx = self.n - 1
#
#         # 用户交互过的物品
#         Iu = set(self.user_train[user])
#         for item in reversed(self.user_train[user][:-1]):
#             seq[idx] = item
#             pos[idx] = nxt
#             # 从为交互过的物品中随机采集一个负样本
#             neg[idx] = np.random.choice(list(self.total_items - Iu), size=1)
#
#             nxt = item
#             idx -= 1
#
#             if idx == -1:
#                 break
#
#         return user, seq, pos, neg

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()