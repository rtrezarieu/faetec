import sys
import lmdb
import torch
import os
import random
import pickle
from tqdm import tqdm
import shutil
import time

def main(dataset_name):
    print(f"Dataset name: {dataset_name}")

    # dataset_name = 'regular_random_3x7x3_20000'
    data_dir = f'data_to_convert/{dataset_name}'
    pred_dir = f'data/{dataset_name}/pred'
    pred_lmdb_path = f'{pred_dir}/pred.mdb'

    os.makedirs(pred_dir, exist_ok=True)

    map_size = 2 * 1024 * 1024 * 1024  # 2 GB

    # Get all .pt files in the data directory
    all_files = [os.path.join(root, filename)
                for root, dirs, files in os.walk(data_dir)
                for filename in files if filename.endswith('.pt')]

    random.shuffle(all_files)
    pred_files = set(all_files)
    print(f"Total files: {len(all_files)}")

    # Initialize LMDB environment
    pred_env = lmdb.open(pred_lmdb_path, map_size=map_size)
    pred_txn = pred_env.begin(write=True)

    pred_counter = 0

    # Iterate through all subdirectories and .pt files and store them in LMDB
    total_files = len(all_files)
    with tqdm(total=total_files, desc="Processing files") as pbar:
        for file_path in all_files:
            data = torch.load(file_path)

            x = data.x
            edge_attr = data.edge_attr
            edge_index = data.edge_index
            y = data.y   ###### can be none, to be implemented

            pos = x[:, 3:6]
            forces = x[:, 9:12]
            beam_col = edge_attr[:, 0:2]

            processed_data = {
                'pos': pos,
                'forces': forces,
                'beam_col': beam_col,
                'edge_index': edge_index,
                'y': y,
            }

            if file_path in pred_files:
                key = f"{pred_counter}".encode("ascii")
                pred_txn.put(key, pickle.dumps(processed_data))
                pred_counter += 1

            pbar.update(1)

    pred_txn.commit()
    pred_env.close()

    # Add a small delay to ensure the environment is fully released
    time.sleep(2)

    # Rename data.mdb to pred.lmdb and move it up one directory
    shutil.move(os.path.join(pred_lmdb_path, 'data.mdb'), os.path.join(pred_dir, 'pred.lmdb'))
    shutil.rmtree(pred_lmdb_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python dataset_convertor.py <dataset_name>")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    # dataset_name = "regular_random_3x7x3_2000"
    main(dataset_name)
