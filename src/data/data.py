import torch
from torch.utils.data import Dataset
from typing import List, Any, Dict
import ipdb
from torch_geometric.datasets import GNNBenchmarkDataset
import torch_geometric.transforms as T

# class MyFilter(object):
#     def __call__(self, data):
#         return data.num_nodes <= max_nodes


class GraphDataset(GNNBenchmarkDataset):
    # doc_list
    # extracted_list as template
    def __init__(self, cfg: Any):
        super(GraphDataset, self).__init__(
            root="/tmp/CIFAR10", name="CIFAR10", transform=T.ToDense(cfg.max_nodes))

    @staticmethod
    def collate_fn(batch):
        x = torch.tensor(batch.x)
        adj = torch.tensor(batch.adj)
        mask = torch.tensor(batch.mask)

        return {
            "x": x,
            "adj": adj,
            "mask": mask
        }
