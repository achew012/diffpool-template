#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 14:47:22 2022

@author: fixitfelix
"""
from math import ceil
import torch
import torch.nn.functional as F
from model.GNN import GNN
from torch_geometric.nn import dense_diff_pool


class DiffPool(torch.nn.Module):
    def __init__(self, cfg):
        super(DiffPool, self).__init__()
        self.cfg = cfg
        num_nodes = ceil(0.25 * cfg.max_nodes)
        #self.gnn1_pool = GNN(dataset.num_features, 64, num_nodes)
        #self.gnn1_embed = GNN(dataset.num_features, 64, 64)

        self.gnn1_pool = GNN(cfg.num_features, 64, num_nodes)
        self.gnn1_embed = GNN(cfg.num_features, 64, 64)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(64, 64, num_nodes)
        self.gnn2_embed = GNN(64, 64, 64, lin=False)
        self.gnn3_embed = GNN(64, 64, 64, lin=False)

        self.lin1 = torch.nn.Linear(64, 64)
        self.lin2 = torch.nn.Linear(64, 64)

    def forward(self, x, adj, mask):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)
        #x_1 = s_0.t() @ z_0
        #adj_1 = s_0.t() @ adj_0 @ s_0

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x, l1 + l2, e1 + e2
