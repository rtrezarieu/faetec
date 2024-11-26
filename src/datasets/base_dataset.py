from pathlib import Path
import sys
import lmdb
import subprocess
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
import pickle

class BaseDataset(Dataset):
    def __init__(self, config, transform=None, fa_frames=None):
        super(BaseDataset, self).__init__()
        self.config = config
        self.path = Path(self.config["src"])

        if not self.path.exists():
            # Eventually add here a code to download the dataset.
            raise FileNotFoundError(f"{self.path} does not exist.")

        self.env = lmdb.open(
            str(self.path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )

        self.num_samples = int(self.env.stat()["entries"])
        self._keys = [f"{i}".encode("ascii") for i in range(self.num_samples)]  # For oc20

        self.transform = transform
        self.fa_frames = fa_frames
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        key = self._keys[idx]
        datapoint_pickle = self.env.begin().get(key)
        if datapoint_pickle is None:
            raise KeyError(f"Key {key} not found in the database.")
        data_object = pickle.loads(datapoint_pickle)
        source = data_object
        data_object =  Data(**{k: v for k, v in source.items() if v is not None})

        if self.transform:
            data_object = self.transform(data_object)

        return data_object
    
    def close_db(self):
        self.env.close()

# For creating the batch from a list of graphs
class ParallelCollater:
    def __init__(self):
        pass

    def __call__(self, data_list):
        batch = Batch.from_data_list(data_list)

        n_neighbors = []
        for _, data in enumerate(data_list):
            n_index = data.edge_index[1, :]
            n_neighbors.append(n_index.shape[0])
        batch.neighbors = torch.tensor(n_neighbors)

        return [batch]