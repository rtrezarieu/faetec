import pickle
import tarfile
import os
import lmdb

def extract_lmdb_archive(archive_path, extract_to):
    print(f"Extracting {archive_path} to {extract_to}")
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(path=extract_to)
    print(f"Extraction complete.")

def print_first_element(lmdb_path):
    # Extract the .tar.gz archive if necessary
    if lmdb_path.endswith('.lmdb'):
        extract_dir = lmdb_path + '_extracted'
        if not os.path.exists(extract_dir):
            extract_lmdb_archive(lmdb_path, extract_dir)
        lmdb_path = os.path.join(extract_dir, 'train.mdb')  # Adjust path to point to the subdirectory

    print(f"Opening LMDB environment at {lmdb_path}")
    env = lmdb.open(lmdb_path, readonly=True)
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            first_element = pickle.loads(value)
            print(f"Key: {key}")
            print(f"Value: {first_element}")
            break

if __name__ == "__main__":
    lmdb_path = 'data/regular_random_gw/train/train.lmdb'
    print_first_element(lmdb_path)

    # Creates a new directory with the extracted contents of the .tar.gz archive