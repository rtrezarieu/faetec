import lmdb
import torch
import os
import random
import pickle
from tqdm import tqdm
import subprocess
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
        beam_col = edge_attr[0:2]

        # Create a dictionary to store in LMDB
        processed_data = {
            'pos': pos,
            'forces': forces,
            'beam_col': beam_col,
            'edge_index': edge_index,
            'y': y,
        }

    

        # Store each structure with a unique key (e.g., filename or index)
        key = os.path.relpath(file_path, data_dir).encode('ascii')  # LMDB keys must be bytes
        value = pickle.dumps(processed_data)  # Serialize the processed data
        
        # Store the key-value pair in LMDB
        # txn.put(key, value)
        if file_path in train_files:
            train_txn.put(key, value)
        elif file_path in val_files:
            val_txn.put(key, value)
        # Update the progress bar
        pbar.update(1)


# Commit the transaction and close the environment
train_txn.commit()
val_txn.commit()
train_env.close()
val_env.close()

# Add a small delay to ensure the environment is fully released
time.sleep(2)

# Package the LMDB directories into tar.gz archives using subprocess
# subprocess.run(['tar', '-czvf', 'train.lmdb', '-C', train_dir, os.path.basename(train_lmdb_path)], check=True)
# shutil.move('train.lmdb', train_dir)
subprocess.run(['tar', '-czvf', os.path.join(train_dir, 'train.lmdb'), '-C', train_dir, 'train.mdb'], check=True)   #os.path.basename(train_lmdb_path)

# subprocess.run(['tar', '-czvf', 'val.lmdb', '-C', val_dir, os.path.basename(val_lmdb_path)], check=True)
# shutil.move('val.lmdb', val_dir)
subprocess.run(['tar', '-czvf', os.path.join(val_dir, 'val.lmdb'), '-C', val_dir, 'val.mdb'], check=True) #os.path.basename(val_lmdb_path)

# Remove the LMDB directories
shutil.rmtree(train_lmdb_path)
shutil.rmtree(val_lmdb_path)