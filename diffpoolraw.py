#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 14:33:44 2022

@author: fixitfelix
"""

import os.path as osp
from collections import Counter
import copy
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch_geometric.datasets import GNNBenchmarkDataset
from tqdm.notebook import tqdm

import DiffPool

max_nodes = 150
k=10
lr=0.001

class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes <= max_nodes


dataset = GNNBenchmarkDataset('data', name='CIFAR10', transform=T.ToDense(max_nodes),
                    pre_filter=MyFilter())
dataset = dataset.shuffle()

n = (len(dataset) + 9) // 10

test_dataset = dataset[:n]
val_dataset = dataset[n:2 * n]
train_dataset = dataset[2 * n:]

test_loader = DenseDataLoader(test_dataset, batch_size=512, num_workers=5)
val_loader = DenseDataLoader(val_dataset, batch_size=512, num_workers=5)
train_loader = DenseDataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=5)

        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DiffPool().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(epoch):
    model.train()
    loss_all = 0

    for _, data in enumerate(tqdm(train_loader, desc="Iteration")):
        data = data.to(device)
        optimizer.zero_grad()
        output, _, _ = model(data.x, data.adj, data.mask)
        sim_mat = cos_sim(output, output)
        loss = torch.nn.MSELoss()(sim_mat, generate_indicator_matrix(data).type_as(output))
        loss.backward()
        loss_all += data.y.size(0) * loss.item()
        # Log train loss here
        optimizer.step()
    return loss_all / len(train_dataset), output
e

@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0

    for _, data in enumerate(tqdm(loader, desc="Iteration")):
        data = data.to(device)
        pred = model(data.x, data.adj, data.mask)[0].max(dim=1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)

# =============================================================================
# @torch.no_grad()
# def eval(model, loader, k=10):
#     model.eval()
#     hit_rate = []
# 
# 
#     for step, batch in enumerate(tqdm(loader, desc="Iteration")):
#         batch = batch.to(device)
# 
#         if batch.x.shape[0] < 30:
#             pass
#         else:
#             with torch.no_grad():
#                 pred = model(batch.x, batch.adj, batch.mask)[0]
#                 pred = cos_sim(pred, pred) # N * N
#                 
#             top_k = torch.topk(pred, k, dim=-1).indices[:,1:]
#             #groundtruth_matrix = batch.y.squeeze(dim=-1).repeat(batch.y.shape[0], 1)
#             #print("gt: ", groundtruth_matrix.shape)
#             groundtruth_matrix = generate_indicator_matrix(batch)
#             selected_graph = torch.gather(groundtruth_matrix, dim=-1, index=top_k)
#             hit_k = selected_graph.sum()
#             #print("hit_k", hit_k)
#             hit_rate.append((hit_k / (ground_truth_count(batch) - batch.y.shape[0])))
#             #print(hit_rate)
#             # batch.y shape: N, 1 is ground truth
#             #extended_groundtruth = batch.y.repeat(1, k+1)
#             
#             #hit_k = ((selected_graph == extended_groundtruth) + 0).sum()
#             #hit_rate.append((hit_k - batch.y.shape[0]) / (batch.y.shape[0] * k))
#         
#     return torch.tensor(hit_rate).mean()
# =============================================================================

best_model = None
best_val_acc = test_acc = 0
for epoch in range(1, 101):
    train_loss, train_pred = train(epoch)
    val_acc = eval(model, val_loader)
    #test_acc = eval(test_loader)
    if val_acc > best_val_acc:
        #test_acc = test(test_loader)
        best_val_acc = val_acc
        best_model = copy.deepcopy(model)
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
          f'Val Acc: {val_acc:.4f}')
