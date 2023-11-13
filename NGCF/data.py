import random

from scipy import sparse
import numpy as np


class Data:
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size

        self.prefix = 'Data/' + dataset + '/'

        self.user_nums, self.item_nums = 0, 0
        self.training_nums, self.test_nums = 0, 0
        self.exist_users = []
        self.load_data()

        # self.Iu = [item for item in range(self.item_nums)]

    def load_data(self):
        # training data
        with open(self.prefix + 'train.txt', 'r') as f:
            for line in f.readlines():
                splited = line.split('\n')[0].split(' ')
                items = [int(item) for item in splited[1:]]
                user = int(splited[0])

                self.item_nums = max(self.item_nums, max(items))
                self.user_nums = max(self.user_nums, user)

                self.exist_users.append(user)
                self.training_nums += len(items)

        # test data
        with open(self.prefix + 'test.txt', 'r') as f:
            for line in f.readlines():
                splited = line.split('\n')[0].split(' ')
                items = [int(item) for item in splited[1:]]
                user = int(splited[0])

                self.test_nums += len(items)
                self.item_nums = max(self.item_nums, max(items))
                self.user_nums = max(self.user_nums, user)

        # true number should plus 1
        self.user_nums += 1
        self.item_nums += 1

        # the sparse matrix based on dict, it can use toarray() to change it into dense array
        self.R = sparse.dok_matrix((self.user_nums, self.item_nums), dtype=float)

        self.train_data, self.test_data = {}, {}

        # filling matrix
        with open(self.prefix + 'train.txt', 'r') as f:
            for line in f.readlines():
                splited = line.split('\n')[0].split(' ')
                items = [int(item) for item in splited[1:]]
                user = int(splited[0])

                # R just for training data interaction
                for item in items:
                    self.R[user, item] = 1

                self.train_data[user] = items

        with open(self.prefix + 'test.txt', 'r') as f:
            for line in f.readlines():
                splited = line.split('\n')[0].split(' ')
                items = [int(item) for item in splited[1:]]
                user = int(splited[0])

                self.test_data[user] = items

    def get_adj_mat(self):
        try:
            adj_mat = sparse.load_npz(self.prefix + 'adj_mat.npz')
            norm_adj_mat = sparse.load_npz(self.prefix + 'norm_adj_mat.npz')
            mean_adj_mat = sparse.load_npz(self.prefix + 'mean_adj_mat.npz')
            print("data already exist!")
        except FileNotFoundError:
            print("data not exist!")
            # variable -> paper symbol
            # adj_mat ~ matrix A
            # norm_adj_mat ~ Laplacian matrix L
            adj_mat, norm_adj_mat, mean_adj_mat = self.creat_adj_mat()
            sparse.save_npz(self.prefix + 'adj_mat.npz', adj_mat)
            sparse.save_npz(self.prefix + 'norm_adj_mat.npz', norm_adj_mat)
            sparse.save_npz(self.prefix + 'mean_adj_mat.npz', mean_adj_mat)

        return adj_mat, norm_adj_mat, mean_adj_mat

    def creat_adj_mat(self):

        adj_mat = sparse.dok_matrix((self.user_nums + self.item_nums, self.user_nums + self.item_nums), dtype=float)
        adj_mat = adj_mat.tolil()

        R = self.R.tolil()

        adj_mat[:self.user_nums, self.user_nums:] = R
        adj_mat[self.user_nums:, :self.user_nums] = R.T
        adj_mat = adj_mat.todok()

        row_sum = np.array(adj_mat.sum(1))

        # create mean adj D^-1 * A, don't know how to use it
        d_inv = np.power(row_sum, -1).flatten()
        # to fill user/item has no interactions(maybe it doesn't exist, just for safe)
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sparse.diags(d_inv)

        mean_adj_mat = d_mat_inv.dot(adj_mat)

        # create norm adj D^-1/2 * A * D^-1/2
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv)] = 0.
        d_mat_inv_sqrt = sparse.diags(d_inv_sqrt)

        norm_adj_mat = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt)

        return adj_mat.tocsr(), norm_adj_mat.tocoo().tocsr(), mean_adj_mat.tocoo().tocsr()

    def sample(self):
        # if user_nums < batch_size, should select duplicated user
        if self.batch_size <= self.user_nums:
            users = random.sample(self.exist_users, self.batch_size)
        else:
            users = [random.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos(user, num):
            pos_items = self.train_data[user]
            sampled_pos_items = []

            for _ in range(num):
                pos_item = random.choice(pos_items)

                while pos_item in sampled_pos_items:
                    pos_item = random.choice(pos_items)

                sampled_pos_items.append(pos_item)

            return sampled_pos_items

        def sample_neg(user, num):
            pos_items = self.train_data[user]
            sampled_neg_items = []

            for _ in range(num):
                # it is too slow?
                # neg_item = random.choice(list(set(self.Iu) - set(pos_items)))
                neg_item_id = np.random.randint(low=0, high=self.item_nums, size=1)[0]

                while neg_item_id in self.train_data[user] and neg_item_id in sampled_neg_items:
                    neg_item_id = np.random.randint(low=0,high=self.item_nums, size=1)[0]

                sampled_neg_items.append(neg_item_id)

            return sampled_neg_items

        pos_items, neg_items = [], []
        # sample a triple (u, i, j) where i belong to positive items and j belong to negative items
        for user in users:
            pos_items.extend(sample_pos(user, 1))
            neg_items.extend(sample_neg(user, 1))

        return users, pos_items, neg_items