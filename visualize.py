import os
import torch
import torch.utils.data
import hydra
from torch_geometric.loader import DataLoader
from omegaconf import DictConfig, OmegaConf
from src.datasets.base_dataset import BaseDataset
from src.utils import visualize_graph_as_3D_structure, visualize_graphs
from src.train import transformations_list


class SimpleDatasetLoader:
    def __init__(self, config):
        self.config = config
        self.load_dataset()
        self.predictions = self.load_predictions()

    def load_dataset(self):
        """Loads the training dataset and prints some basic information."""
        self.transform = self._get_transform()
        # Assuming BaseDataset is a class that you use for handling your dataset
        if 'train' in self.config['dataset']:
            self.dataset = BaseDataset(self.config['dataset']['train'], transform=None)  ##################### NONE   self.transform
        else:
            self.dataset = BaseDataset(self.config['dataset']['pred'], transform=None)  ##################### NONE   self.transform

        print(f"Loaded dataset with {len(self.dataset)} samples.")

    def _get_transform(self):
        """Returns transformations if needed."""
        return transformations_list(self.config) if 'transform' in self.config else None

    def get_single_sample_loader(self, index=0):
        """Get a specific sample from the dataset."""
        if index >= len(self.dataset):
            raise IndexError("Sample index out of range.")
        
        single_sample = torch.utils.data.Subset(self.dataset, [index])
        return DataLoader(single_sample, batch_size=1, shuffle=False)  # torch.utils.data.
    
    def load_predictions(self):
        """Loads saved predictions from disk."""
        preds_save_path = self.config["dataset"].get("save_preds_path", "predictions/saved_predictions.pth")
        if os.path.exists(preds_save_path):
            predictions = torch.load(preds_save_path)
            print(f"Predictions loaded from {preds_save_path}")
            return predictions
        else:
            print(f"No saved predictions found at {preds_save_path}")
            return None

    def get_sample_predictions(self, index=0):
        """Fetch predictions for a specific sample."""
        if self.predictions is None:
            print("No predictions available. Please generate or load predictions.")
            return None

        if index >= len(self.predictions):
            raise IndexError("Sample index out of range.")

        return self.predictions['disp']


@hydra.main(config_path="configs", config_name="default_config.yaml", version_base="1.1")
def main(config: DictConfig):
    disp_scaling = 100
    dataset_loader = SimpleDatasetLoader(config)
    single_sample_loader = dataset_loader.get_single_sample_loader(index=0) ###### index is only useful if we want to check one structure from a train, without predictions
    prediction = dataset_loader.get_sample_predictions()


    # (only one sample here)
    for sample_data in single_sample_loader:
        x_list = sample_data.pos[:, 0].tolist()
        y_list = sample_data.pos[:, 1].tolist()  # reversed for matplotlib xyz convention vs xzy for pythagore
        z_list = sample_data.pos[:, 2].tolist()

        x_target_list = [x + disp_scaling * y for x, y in zip(x_list, sample_data.y[:, 0].tolist())]
        y_target_list = [x + disp_scaling * y for x, y in zip(y_list, sample_data.y[:, 1].tolist())]
        z_target_list = [x + disp_scaling * y for x, y in zip(z_list, sample_data.y[:, 2].tolist())]

        x_pred_list = [x + disp_scaling * y for x, y in zip(x_list, prediction[:, 0].tolist())]
        y_pred_list = [x + disp_scaling * y for x, y in zip(y_list, prediction[:, 1].tolist())]
        z_pred_list = [x + disp_scaling * y for x, y in zip(z_list, prediction[:, 2].tolist())]
        
        # visualize_graph_as_3D_structure(sample_data, x_list, y_list, z_list, color='k')
        # visualize_graph_as_3D_structure(sample_data, x_target_list, y_target_list, z_target_list, color='r')
        
        visualize_graphs(sample_data, x_list, y_list, z_list, x_target_list, y_target_list, z_target_list, 
                         x_pred_list, y_pred_list, z_pred_list, show_vectors=True)
        ### change colors
        ### change nodes color
        ### print predictions
        ### compare target and predictions
        ### option to print the vector only

if __name__ == "__main__":
    main()