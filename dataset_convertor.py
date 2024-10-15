import lmdb
import torch
import os
import random
import pickle
from tqdm import tqdm
import shutil
import time

dataset_name = 'regular_random_gw'
data_dir = f'data_to_convert/{dataset_name}'
train_dir = f'data/{dataset_name}/train'
val_dir = f'data/{dataset_name}/val'
train_lmdb_path = f'{train_dir}/train.mdb'
val_lmdb_path = f'{val_dir}/val.mdb'

# Ensure train and val directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Set the map size to a large enough value (adjust this based on dataset size)
map_size = 2 * 1024 * 1024 * 1024  # 2 GB

train_split_ratio = 0.9

# Get all .pt files in the data directory
all_files = [os.path.join(root, filename)
             for root, dirs, files in os.walk(data_dir)
             for filename in files if filename.endswith('.pt')]

# Shuffle the files to ensure random distribution
random.shuffle(all_files)

# Calculate the split index
split_index = int(train_split_ratio * len(all_files))

# Split the files into train and val sets
train_files = set(all_files[:split_index])
val_files = set(all_files[split_index:])

# Print the number of files in each set for verification
print(f"Total files: {len(all_files)}")
print(f"Train files: {len(train_files)}")
print(f"Validation files: {len(val_files)}")

# Initialize LMDB environment
train_env = lmdb.open(train_lmdb_path, map_size=map_size)
val_env = lmdb.open(val_lmdb_path, map_size=map_size)
train_txn = train_env.begin(write=True)  # Begin the transaction for writing
val_txn = val_env.begin(write=True)  # Begin the transaction for writing

# Initialize counters for train and validation keys
train_counter = 0
val_counter = 0

# Iterate through all subdirectories and .pt files and store them in LMDB
total_files = len(all_files)
with tqdm(total=total_files, desc="Processing files") as pbar:
    for file_path in all_files:
        data = torch.load(file_path)

        x = data.x
        edge_attr = data.edge_attr
        edge_index = data.edge_index
        y = data.y

        pos = x[:, 3:6]
        forces = x[:, 9:12]
        beam_col = edge_attr[:, 0:2]

        # Create a dictionary to store in LMDB
        processed_data = {
            'pos': pos,
            'forces': forces,
            'beam_col': beam_col,
            'edge_index': edge_index,
            'y': y,
        }

        # Store each structure with a unique key based on train/val split
        if file_path in train_files:
            key = f"{train_counter}".encode("ascii")  # Use the train counter for train files
            train_txn.put(key, pickle.dumps(processed_data))  # Serialize and store in LMDB
            train_counter += 1  # Increment train counter
        elif file_path in val_files:
            key = f"{val_counter}".encode("ascii")  # Use the val counter for val files
            val_txn.put(key, pickle.dumps(processed_data))  # Serialize and store in LMDB
            val_counter += 1  # Increment val counter

        # Update the progress bar
        pbar.update(1)

# Commit the transaction and close the environment
train_txn.commit()
val_txn.commit()
train_env.close()
val_env.close()

# Add a small delay to ensure the environment is fully released
time.sleep(2)

# Rename data.mdb to train.lmdb and move it up one directory
shutil.move(os.path.join(train_lmdb_path, 'data.mdb'), os.path.join(train_dir, 'train.lmdb'))
shutil.move(os.path.join(val_lmdb_path, 'data.mdb'), os.path.join(val_dir, 'val.lmdb'))
shutil.rmtree(train_lmdb_path)
shutil.rmtree(val_lmdb_path)