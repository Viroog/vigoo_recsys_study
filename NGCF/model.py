import torch
import torch.nn as nn
import scipy.sparse as sparse
import torch.nn.functional as F
import numpy as np

class NGCF(nn.Module):
    def __init__(self, user_nums, item_nums, norm_adj_mat, args):
        super(NGCF, self).__init__()
        self.user_nums = user_nums
        self.item_nums = item_nums

        # the original laplacian matrix, that is symbol L in paper
        self.laplacian_mat = norm_adj_mat
        # laplacian matrix + I, where I is diagonal matrix
        self.adj_mat = norm_adj_mat + sparse.eye(norm_adj_mat.shape[0])

        self.embed_size = args.embed_size
        self.layers = eval(args.layer_size)

        self.args = args

        self.init_weight()
        # self.my_init_weight()
        self.L = self.convert_sparse_mat_to_sparse_tensor(self.laplacian_mat)
        self.A = self.convert_sparse_mat_to_sparse_tensor(self.adj_mat)

    # use my own way to implement initialize
    def my_init_weight(self):

        self.user_embedding = nn.Embedding(self.user_nums, self.embed_size)
        self.item_embedding = nn.Embedding(self.item_nums, self.embed_size)

        self.weight_dict = nn.ParameterDict()
        layers = [self.embed_size] + self.layers

        for layer in range(len(self.layers)):
            # liner obtain both W and b
            self.weight_dict.update({
                f'W1_{layer}': nn.Linear(layers[layer], layers[layer + 1])
            })

            self.weight_dict.update({
                f'W2_{layer}': nn.Linear(layers[layer], layers[layer + 1])
            })

        for name, param in self.named_parameters():
            try:
                nn.init.xavier_uniform_(param.data)
            # ignore the layer that fail to initialize
            except Exception:
                pass

    def init_weight(self):
        # a new way to initialize parameter
        initializer = nn.init.xavier_uniform_

        self.embedding_dict = nn.ParameterDict({
            'user_embed': nn.Parameter(initializer(torch.empty(self.user_nums, self.embed_size))),
            'item_embed': nn.Parameter(initializer(torch.empty(self.item_nums, self.embed_size)))
        })

        # can these weights use liner layer to replace?
        self.weight_dict = nn.ParameterDict()
        layers = [self.embed_size] + self.layers
        # each layer has 4 parameters: W1 b1 W2 b2
        for layer in range(len(self.layers)):
            # W1 to transform neighbors
            self.weight_dict.update({
                f'W1_{layer}': nn.Parameter(initializer(torch.empty(layers[layer], layers[layer + 1])))
            })

            # b1 for W1
            self.weight_dict.update({
                f'b1_{layer}': nn.Parameter(initializer(torch.empty(1, layers[layer + 1])))
            })

            # W2 to transform the affinity between itself and neighbor
            self.weight_dict.update({
                f'W2_{layer}': nn.Parameter(initializer(torch.empty(layers[layer], layers[layer + 1])))
            })

            # b2 for W2
            self.weight_dict.update({
                f'b2_{layer}': nn.Parameter(initializer(torch.empty(1, layers[layer + 1])))
            })

    def convert_sparse_mat_to_sparse_tensor(self, mat):
        coo = mat.tocoo()
        # ijv format
        # the source code is extremely slow
        ij = torch.LongTensor(np.mat([coo.row, coo.col]))
        v = torch.FloatTensor(coo.data)

        sparse_mat = torch.sparse.FloatTensor(ij, v, coo.shape).to(self.args.device)

        return sparse_mat

    def dropout_sparse(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        # torch.rand generate numbers between 0 and 1, plus 1 - rate, equal (1-rate, 2-rate)
        # and all number smaller than 1 will be dropout, that is dropout rate is rate
        random_tensor += torch.rand(noise_shape).to(self.args.device)
        mask = torch.floor(random_tensor).type(torch.bool)

        indices = x._indices()
        values = x._values()

        indices = indices[:, mask]
        values = values[mask]

        out = torch.sparse.FloatTensor(indices, values, x.shape).to(self.args.device)

        # to insure the variance will not be change
        return out * (1. / (1 - rate))

    def get_rating(self, user_embedding, item_embedding):
        return torch.sum(torch.mul(user_embedding, item_embedding), dim=1)

    def get_bpr_loss(self, user_embedding, pos_item_embedding, neg_item_embedding):
        # shape: (1024, )
        pos_rating = self.get_rating(user_embedding, pos_item_embedding)
        neg_rating = self.get_rating(user_embedding, neg_item_embedding)

        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_rating - neg_rating)))

        # regulization
        l2norm = (torch.norm(user_embedding) ** 2 +
                  torch.norm(pos_item_embedding) ** 2 +
                  torch.norm(neg_item_embedding)) / 2.
        l2reg = self.args.l2_reg * l2norm / user_embedding.shape[0]

        return bpr_loss + l2reg

    def forward(self, users, pos_items, neg_items):

        # node dropout
        # type can be coalesced/uncoalesced(uncoalesced in same indices can have more than one value), if type is uncoalesced it should add _ when refre to the its attribute
        L_hat = self.dropout_sparse(self.L, self.args.node_dropout_rate,
                                    self.L._nnz()) if self.args.node_dropout else self.L
        A_hat = self.dropout_sparse(self.A, self.args.node_dropout_rate, ) if self.args.node_dropout else self.A
        # print(self.A.is_coalesced())
        # print(self.L.is_coalesced())

        embedding_list = []
        # the embedding concat user embedding and item embedding
        # shape: (user_nums+item_nums, embed_size)
        ui_embedding = torch.cat([self.embedding_dict['user_embed'], self.embedding_dict['item_embed']], dim=0)
        embedding_list.append(ui_embedding)

        # L_hat and A_hat matrix is torch.sparse
        for layer in range(len(self.layers)):
            # shape: (user_nums+item_nums, layers[layer])
            A_mul_embedding = torch.sparse.mm(A_hat, ui_embedding)
            L_mul_embedding = torch.sparse.mm(L_hat, ui_embedding)

            # the result on the plus symbol left in equal(7)
            # shape: (user_nums+item_nums, layers[layer+1])
            left_result = torch.matmul(A_mul_embedding, self.weight_dict[f'W1_{layer}']) + self.weight_dict[
                f'b1_{layer}']

            # shape: (user_nums+item_nums, layers[layer])
            affinity = torch.mul(L_mul_embedding, ui_embedding)

            # shape: (user_nums+item_nums, layers[layer+1])
            right_result = torch.matmul(affinity, self.weight_dict[f'W2_{layer}']) + self.weight_dict[f'b2_{layer}']

            ui_embedding = nn.LeakyReLU(negative_slope=0.2)(left_result + right_result)

            ui_embedding = nn.Dropout(p=self.args.message_dropout_rate)(ui_embedding)

            # normalization(this step not mention in the paper)
            normed_ui_embedding = F.normalize(ui_embedding, p=2, dim=1)

            embedding_list.append(normed_ui_embedding)

        all_embedding = torch.cat(embedding_list, dim=1)

        user_embedding = all_embedding[torch.LongTensor(users).to(self.args.device)]
        pos_item_embedding = all_embedding[torch.LongTensor(pos_items).to(self.args.device)]
        neg_item_embedding = all_embedding[torch.LongTensor(neg_items).to(self.args.device)]

        # final shape: (batch_size, embed_size+layers[0~3])
        # for i in range(len(embedding_list)):
        #     if i == 0:
        #         # this is no embedding layer, should use [] to get slice not ()
        #         user_embedding = embedding_list[i][torch.LongTensor(users).to(self.args.device)]
        #         pos_item_embedding = embedding_list[i][torch.LongTensor(pos_items).to(self.args.device)]
        #         neg_item_embedding = embedding_list[i][torch.LongTensor(neg_items).to(self.args.device)]
        #     else:
        #         user_embedding = torch.cat(
        #             [user_embedding, embedding_list[i][torch.LongTensor(users).to(self.args.device)]], dim=1)
        #         pos_item_embedding = torch.cat(
        #             [pos_item_embedding, embedding_list[i][torch.LongTensor(pos_items).to(self.args.device)]], dim=1)
        #         neg_item_embedding = torch.cat(
        #             [neg_item_embedding, embedding_list[i][torch.LongTensor(neg_items).to(self.args.device)]], dim=1)

        loss = self.get_bpr_loss(user_embedding, pos_item_embedding, neg_item_embedding)

        return loss
