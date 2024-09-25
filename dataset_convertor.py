import lmdb
import torch
import os
import pickle
from tqdm import tqdm

dataset_name = 'regular_random'
data_dir = f'data/{dataset_name}'
lmdb_path = f'{data_dir}/{dataset_name}.lmdb'

# Set the map size to a large enough value (you can adjust this based on dataset size)
map_size = 2 * 1024 * 1024 * 1024  # 2 GB

# Initialize LMDB environment
env = lmdb.open(lmdb_path, map_size=map_size)
txn = env.begin(write=True)  # Begin the transaction for writing

# Iterate through all subdirectories and .pt files and store them in LMDB
# Get the total number of .pt files for a single progress bar
total_files = sum([len(files) for r, d, files in os.walk(data_dir) if any(f.endswith('.pt') for f in files)])
with tqdm(total=total_files, desc="Processing files") as pbar:
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith('.pt'):
                file_path = os.path.join(root, filename)
                data = torch.load(file_path)
                
                x = data.x
                edge_attr = data.edge_attr
                edge_index = data.edge_index
                y = data.y

                # Create a dictionary to store in LMDB
                processed_data = {
                    'x': x,
                    'edge_attr': edge_attr,
                    'edge_index': edge_index,
                    'y': y,
                }

                # Store each structure with a unique key (e.g., filename or index)
                key = os.path.relpath(file_path, data_dir).encode('ascii')  # LMDB keys must be bytes
                value = pickle.dumps(processed_data)  # Serialize the processed data
                
                # Store the key-value pair in LMDB
                txn.put(key, value)
                
                # Update the progress bar
                pbar.update(1)


# Commit the transaction and close the environment
txn.commit()
env.close()

# Package the LMDB directory
# tar -czvf regular_random.lmdb.tar.gz regular_random.lmdb/
