import os
import yaml
import torch
import pickle
from tqdm import tqdm

# Paths to the dataset files
dataset_name = 'regular_random_gw'
data_dir = f'data_to_convert/{dataset_name}'
config_path = f'configs/dataset/structures.yaml'

all_files = [os.path.join(root, filename)
             for root, dirs, files in os.walk(data_dir)
             for filename in files if filename.endswith('.pt')]

# Accumulators for disp, N, and M
all_disp = []
all_N = []
all_M = []

# Iterate through all .pt files
total_files = len(all_files)
with tqdm(total=total_files, desc="Processing files") as pbar:
    for file_path in all_files:
        data = torch.load(file_path)
        y = data.y

        # Extract components
        disp = y[:, 0:3]
        N = torch.cat([y[:, 3:9], y[:, 9:15], y[:, 15:21]], dim=-1)
        M = torch.cat([y[:, 21:27], y[:, 27:33], y[:, 33:39]], dim=-1)

        # Accumulate data
        all_disp.append(disp)
        all_N.append(N)
        all_M.append(M)

        # Update the progress bar
        pbar.update(1)

# Concatenate all accumulated data
all_disp = torch.cat(all_disp, dim=0)
all_N = torch.cat(all_N, dim=0)
all_M = torch.cat(all_M, dim=0)

# Calculate standard deviations
std_disp = torch.std(all_disp)
std_N = torch.std(all_N)
std_M = torch.std(all_M)

# Configuration dictionary
config = {
    'name': 'structures',
    'train': {
        'src': 'data/regular_random_gw/train/train.lmdb',
        'normalize_labels': True,
        'target_mean_disp': 0,
        'target_std_disp': std_disp.item(),
        'target_mean_N': 0,
        'target_std_N': std_N.item(),
        'target_mean_M': 0,
        'target_std_M': std_M.item(),
    },
    'val': {
        'val': {
            'src': 'data/regular_random_gw/val/val.lmdb',
        }
    },
    'oc20': False
}

# Create directory if it doesn't exist
os.makedirs(os.path.dirname(config_path), exist_ok=True)

# Write configuration to YAML file
with open(config_path, 'w') as file:
    yaml.dump(config, file, default_flow_style=False)

print(f"Configuration file saved to {config_path}")