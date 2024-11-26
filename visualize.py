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
        self.predictions = self.load_predictions(epoch=4)
        self.transformations = self.load_predictions(transformations=True)
        self.transformed_base = self.load_predictions(base=True)

    def load_dataset(self):
        """Loads the training dataset and prints some basic information."""
        self.transform = self._get_transform()
        if 'train' in self.config['dataset']:
            self.dataset = BaseDataset(self.config['dataset']['val'], transform=None)
        else:
            self.dataset = BaseDataset(self.config['dataset']['pred'], transform=None)

        print(f"Loaded dataset with {len(self.dataset)} samples.")

    def _get_transform(self):
        """Returns transformations if needed."""
        return transformations_list(self.config) if 'transform' in self.config else None

    def get_single_sample_loader(self, index=0):
        """Get a specific sample from the dataset."""
        if index >= len(self.dataset):
            raise IndexError("Sample index out of range.")
        
        single_sample = torch.utils.data.Subset(self.dataset, [index])
        return DataLoader(single_sample, batch_size=1, shuffle=False)
    
    def load_predictions(self, epoch=4, transformations=False, base=False):
        """Loads saved predictions from disk."""
        if transformations:
            preds_save_path = self.config["dataset"].get("save_transformed_preds_path", "transformations/saved_transformations.pth")
        elif base:
            preds_save_path = self.config["dataset"].get("save_transformed_base_path", "transformations/saved_base.pth")
        else:
            preds_save_path = self.config["dataset"].get("save_preds_path", "predictions/saved_predictions.pth")
        if os.path.exists(preds_save_path):
            if preds_save_path.endswith('.pt'): # for predict.py
                predictions = torch.load(preds_save_path)
                print(f"Predictions loaded from {preds_save_path}")
            else: # for train.py
                preds_save_path = os.path.join(preds_save_path, f"epoch_{epoch}.pt")
            predictions = torch.load(preds_save_path)
            return predictions
        else:
            print(f"No saved predictions found at {preds_save_path}")
            return None
    

    def get_sample_predictions(self, index=0):
        """Fetch predictions for a specific sample."""
        preds, transformations, transformed_base = None, None, None
        if self.predictions:
            preds = self.predictions['disp']
        if self.transformations:
            transformations = self.transformations['disp']
        if self.transformed_base:
            transformed_base = self.transformed_base['base']
        else:
            print("No predictions available. Please generate or load predictions.")
            return None
        return preds, transformations, transformed_base


@hydra.main(config_path="configs", config_name="default_config.yaml", version_base="1.1")
def main(config: DictConfig):
    # For sweeper multirun
    original_cwd = hydra.utils.get_original_cwd()
    os.chdir(original_cwd)

    disp_scaling = 20
    dataset_loader = SimpleDatasetLoader(config)
    # index of the structure in val to visualize. Works only with index=0 for now, because the index is never specified in prediction (etc.), so zip handles it well and select only the first elements.
    single_sample_loader = dataset_loader.get_single_sample_loader(index=0) ###### index is only useful if we want to check one structure from a train, without predictions
    prediction, transformation, transformed_base = dataset_loader.get_sample_predictions()

    # (only one sample here)
    for sample_data in single_sample_loader:
        x_list = sample_data.pos[:, 0].tolist()
        y_list = sample_data.pos[:, 1].tolist()  # reversed for matplotlib xyz convention vs xzy for pythagore
        z_list = sample_data.pos[:, 2].tolist()

        x_forces_list = sample_data.forces[:, 0].tolist()
        y_forces_list = sample_data.forces[:, 1].tolist()
        z_forces_list = sample_data.forces[:, 2].tolist()

        supports_list = sample_data.supports.tolist()

        x_target_list = [x + disp_scaling * y for x, y in zip(x_list, sample_data.y[:, 0].tolist())]
        y_target_list = [x + disp_scaling * y for x, y in zip(y_list, sample_data.y[:, 1].tolist())]
        z_target_list = [x + disp_scaling * y for x, y in zip(z_list, sample_data.y[:, 2].tolist())]

        if prediction is not None:
            x_pred_list = [x + disp_scaling * y for x, y in zip(x_list, prediction[:, 0].tolist())]
            y_pred_list = [x + disp_scaling * y for x, y in zip(y_list, prediction[:, 1].tolist())]
            z_pred_list = [x + disp_scaling * y for x, y in zip(z_list, prediction[:, 2].tolist())]
        else:
            x_pred_list = None
            y_pred_list = None
            z_pred_list = None
        
        if transformed_base is not None:
            x_transf_base_list = [x for x in transformed_base[:, 0].tolist()]
            y_transf_base_list = [x for x in transformed_base[:, 1].tolist()]
            z_transf_base_list = [x for x in transformed_base[:, 2].tolist()]
        else:
            x_transf_base_list = None
            y_transf_base_list = None
            z_transf_base_list = None

        if transformation is not None:    
            x_transf_list = [x + disp_scaling * y for x, y in zip(x_transf_base_list, transformation[:, 0].tolist())]
            y_transf_list = [x + disp_scaling * y for x, y in zip(y_transf_base_list, transformation[:, 1].tolist())]
            z_transf_list = [x + disp_scaling * y for x, y in zip(z_transf_base_list, transformation[:, 2].tolist())]
        else:
            x_transf_list = None
            y_transf_list = None
            z_transf_list = None
        
        visualize_graphs(
                        sample_data, x_list, y_list, z_list, x_target_list, y_target_list, z_target_list, 
                        x_pred_list, y_pred_list, z_pred_list, x_forces_list, y_forces_list, z_forces_list, supports_list,
                        x_transf_list, y_transf_list, z_transf_list, x_transf_base_list, y_transf_base_list, z_transf_base_list
                        )

if __name__ == "__main__":
    main()