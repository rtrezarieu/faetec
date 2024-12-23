import os
import sys
import yaml
import torch
from tqdm import tqdm

def main(dataset_name):
    data_dir = f'data_to_convert/{dataset_name}'
    config_path = f'configs/dataset/{dataset_name}.yaml'

    all_files = [os.path.join(root, filename)
                for root, dirs, files in os.walk(data_dir)
                for filename in files if filename.endswith('.pt')]

    all_disp = []
    all_N = []
    all_M = []

    # Iterate through all .pt files
    total_files = len(all_files)
    with tqdm(total=total_files, desc="Processing files") as pbar:
        for file_path in all_files:
            data = torch.load(file_path)
            y = data.y

            disp = y[:, 0:3]
            N = torch.cat([y[:, 3:9], y[:, 9:15], y[:, 15:21]], dim=-1)
            M = torch.cat([y[:, 21:27], y[:, 27:33], y[:, 33:39]], dim=-1)

            all_disp.append(disp)
            all_N.append(N)
            all_M.append(M)

            pbar.update(1)

    all_disp = torch.cat(all_disp, dim=0)
    all_N = torch.cat(all_N, dim=0)
    all_M = torch.cat(all_M, dim=0)

    std_disp = torch.std(all_disp)
    std_N = torch.std(all_N)
    std_M = torch.std(all_M)

    config = {
        'name': f'{dataset_name}',
        'save_model_path': f'models/{dataset_name}/model_{dataset_name}.pth',
        'save_preds_path': f'models/{dataset_name}/val/',
        'save_transformed_preds_path': f'models/{dataset_name}/val/transformed_predictions/',
        'save_transformed_base_path': f'models/{dataset_name}/val/transformed_base/',
        'train': {
            'src': f'data/{dataset_name}/train/train.lmdb',
            'normalize_labels': True,
            'target_mean_disp': 0,
            'target_std_disp': std_disp.item(),
            'target_mean_N': 0,
            'target_std_N': std_N.item(),
            'target_mean_M': 0,
            'target_std_M': std_M.item(),
        },
        'val': {
            'src': f'data/{dataset_name}/val/val.lmdb',
            'val': {
                'src': f'data/{dataset_name}/val/val.lmdb',
            }
        },
        'oc20': False
    }

    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    print(f"Configuration file saved to {config_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python config_generator.py <dataset_name>")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    main(dataset_name)