import torch
from torch.utils.data import Dataset
from typing import List, Any, Dict
import ipdb
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset


class GraphDataset(Dataset):
    # doc_list
    # extracted_list as template
    def __init__(self, dataset: List[Dict], tokenizer: Any, cfg: Any):
        self.tokenizer = tokenizer


    def __len__(self):
        """Returns length of the dataset"""
        return len(self.processed_dataset["docid"])

    def __getitem__(self, idx):
        """Gets an example from the dataset. The input and output are tokenized and limited to a certain seqlen."""
        # item = {key: val[idx] for key, val in self.processed_dataset["encodings"].items()}
        item = {'input_ids': self.processed_dataset["input_ids"][idx]}
        return item

    @ staticmethod
    def collate_fn(batch):
        """
        Groups multiple examples into one batch with padding and tensorization.
        The collate function is called by PyTorch DataLoader
        """

        docids = [ex['docid'] for ex in batch]

        return {
            'docid': docids,
        }

    @ staticmethod
    def collate_inference_fn(batch):
        """
        Groups multiple examples into one batch with padding and tensorization.
        The collate function is called by PyTorch DataLoader
        """

        docids = [ex['docid'] for ex in batch]

        return {
            'docid': docids,
        }
