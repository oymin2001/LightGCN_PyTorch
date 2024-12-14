

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class LightGCN(nn.Module):
    def __init__(self, data, hyper_params, device):
        super(LightGCN, self).__init__()
        self.data = data
        self.n_users = data.n_users
        self.n_items = data.n_items
        self.device = device
        self.norm_adj_mat_sparse_tensor = self._COO_to_torch_sparse().to(self.device)

        # hyper-parameters
        self.n_layers = hyper_params["n_layers"]
        self.latent_dim = hyper_params["latent_dim"]

        # Trainable parameters initialized by Xavier Uniform
        self.E_0 = nn.Embedding(self.n_users + self.n_items, self.latent_dim).to(self.device)
        nn.init.xavier_uniform_(self.E_0.weight)

    def _COO_to_torch_sparse(self):
        norm_adj_mat_coo = self.data.create_norm_adj_mat().tocoo().astype(np.float32)

        indices = np.vstack((norm_adj_mat_coo.row, norm_adj_mat_coo.col))
        values = norm_adj_mat_coo.data

        i = torch.LongTensor(indices).to(self.device)
        v = torch.FloatTensor(values).to(self.device)
        shape = norm_adj_mat_coo.shape

        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    def LightGraphConvolution_operation(self):
        all_layer_embedding = [self.E_0.weight]
        E_lyr = self.E_0.weight

        for layer in range(self.n_layers):
            E_lyr = torch.sparse.mm(self.norm_adj_mat_sparse_tensor, E_lyr)
            all_layer_embedding.append(E_lyr)

        all_layer_embedding = torch.stack(all_layer_embedding)
        mean_layer_embedding = torch.mean(all_layer_embedding, axis = 0)

        final_user_Embed, final_item_Embed = torch.split(mean_layer_embedding, [self.n_users, self.n_items])
        initial_user_Embed, initial_item_Embed = torch.split(self.E_0.weight, [self.n_users, self.n_items])

        return final_user_Embed, final_item_Embed, initial_user_Embed, initial_item_Embed

    def forward(self, users, pos_items, neg_items):
        final_user_Embed, final_item_Embed, initial_user_Embed, initial_item_Embed = self.LightGraphConvolution_operation()

        users_emb, pos_emb, neg_emb = final_user_Embed[users], final_item_Embed[pos_items], final_item_Embed[neg_items]
        userEmb0,  posEmb0, negEmb0 = initial_user_Embed[users], initial_item_Embed[pos_items], initial_item_Embed[neg_items]

        return users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0



def bpr_loss(users, users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0):
    pos_scores = torch.mul(users_emb, pos_emb)
    pos_scores = torch.sum(pos_scores, dim=1)
    neg_scores = torch.mul(users_emb, neg_emb)
    neg_scores = torch.sum(neg_scores, dim=1)

    loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

    reg_loss = (1/2)*(userEmb0.norm().pow(2) + posEmb0.norm().pow(2)  + negEmb0.norm().pow(2))/float(len(users)) # L2 norm

    return loss, reg_loss
