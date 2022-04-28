import torch
from torch.utils.data import Dataset
from typing import List, Any, Dict
import ipdb
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset


class GraphDataset(TUDataset):
    # doc_list
    # extracted_list as template
    def __init__(self, cfg: Any):
        super(GraphDataset, self).__init__(
            root="/tmp/ENZYMES", name="ENZYMES", use_node_attr=True)

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
