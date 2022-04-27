#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 14:46:20 2022

@author: fixitfelix
"""
import torch
from torch_geometric.nn import DenseGCNConv as GCNConv
from torch_geometric.nn import BatchNorm


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super(GNN, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        # (32, 150, 3)
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize))
        # (32, 150, 64)
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels)) # 64 (N, C, L)
        
        self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        self.convs.append(GCNConv(hidden_channels, out_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(out_channels))


    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()
        
        for step in range(len(self.convs)):
            #x = self.bns[step](F.relu(self.convs[step](x, adj, mask)))
            x = F.relu(self.convs[step](x, adj, mask))
            x = self.bns[step](x.transpose(1,2)).transpose(1,2)        
        return x
