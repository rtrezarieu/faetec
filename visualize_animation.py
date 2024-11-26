import sys
import os
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torch_geometric.loader import DataLoader
from omegaconf import DictConfig
import hydra
from src.datasets.base_dataset import BaseDataset
from src.utils import visualize_graphs, visualize_graphs_for_animation
from src.train import transformations_list
from matplotlib.animation import FFMpegWriter


class SimpleDatasetLoader:
    def __init__(self, config):
        self.config = config
        self.load_dataset()
        self.predictions = self.load_predictions()
        self.all_predictions = self.load_all_predictions()

    def load_dataset(self):
        """Loads the training dataset and prints some basic information."""
        self.transform = self._get_transform()
        self.dataset = BaseDataset(self.config['dataset']['val'], transform=None)
        print(f"Loaded dataset with {len(self.dataset)} samples.")

    def _get_transform(self):
        """Returns transformations if needed."""
        return transformations_list(self.config) if 'transform' in self.config else None

    def get_single_sample_loader(self, index=0):
        """Get a specific sample from the dataset."""
        single_sample = torch.utils.data.Subset(self.dataset, [index])
        return DataLoader(single_sample, batch_size=1, shuffle=False)

    def load_predictions(self, epoch=14):
        """Loads saved predictions from disk."""
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

    def load_all_predictions(self):
        """Loads predictions for all epochs to optimize animation."""
        all_predictions = {}
        max_epochs = 100
        preds_save_path = self.config["dataset"].get("save_preds_path", "predictions")
        for epoch in range(max_epochs):
            epoch_path = os.path.join(preds_save_path, f"epoch_{epoch}.pt")
            if os.path.exists(epoch_path):
                all_predictions[epoch] = torch.load(epoch_path)
        return all_predictions

    def create_animation(self, sample_data, x_list, y_list, z_list, x_target_list, y_target_list, z_target_list,
                        x_forces_list, y_forces_list, z_forces_list, supports_list, num_epochs=15):
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        fig.set_size_inches(8, 8)

        all_x = x_list + x_target_list
        all_y = y_list + y_target_list
        all_z = z_list + z_target_list
        x_limits = (-5 + min(all_x), max(all_x))
        y_limits = (min(all_y), max(all_y))
        z_limits = (min(all_z) - 5, max(all_z))

        def update_plot(epoch):
            ax.cla()  # Clear the plot to avoid overlaps
            ax.set_title(f"Epoch {epoch}")
            
            prediction = self.all_predictions.get(epoch, None)
            if prediction is not None:
                x_pred_list = [x + 100 * y for x, y in zip(x_list, prediction['disp'][:, 0].tolist())]
                y_pred_list = [x + 100 * y for x, y in zip(y_list, prediction['disp'][:, 1].tolist())]
                z_pred_list = [x + 100 * y for x, y in zip(z_list, prediction['disp'][:, 2].tolist())]
            else:
                x_pred_list = None
                y_pred_list = None
                z_pred_list = None

            # Use the simplified plotting function for animation
            visualize_graphs_for_animation(ax, sample_data, x_list, y_list, z_list, x_target_list, y_target_list, z_target_list,
                                        x_pred_list, y_pred_list, z_pred_list, x_forces_list, y_forces_list,
                                        z_forces_list, supports_list, x_limits, y_limits, z_limits)

        anim = FuncAnimation(fig, update_plot, frames=range(num_epochs), interval=300, repeat=True)

        writer = FFMpegWriter(fps=2, metadata={'artist': 'Me'}, bitrate=1800)
        try:
            anim.save("predictions_animation.mp4", writer=writer, dpi=200)
            print("Animation saved successfully!")
        except Exception as e:
            print(f"Error saving animation: {e}")

        plt.show()



@hydra.main(config_path="configs", config_name="default_config.yaml", version_base="1.1")
def main(config: DictConfig):
    original_cwd = hydra.utils.get_original_cwd()
    os.chdir(original_cwd)

    dataset_loader = SimpleDatasetLoader(config)
    single_sample_loader = dataset_loader.get_single_sample_loader(index=0)

    for sample_data in single_sample_loader:
        x_list = sample_data.pos[:, 0].tolist()
        y_list = sample_data.pos[:, 1].tolist()
        z_list = sample_data.pos[:, 2].tolist()

        x_forces_list = sample_data.forces[:, 0].tolist()
        y_forces_list = sample_data.forces[:, 1].tolist()
        z_forces_list = sample_data.forces[:, 2].tolist()

        supports_list = sample_data.supports.tolist()

        x_target_list = [x + 100 * y for x, y in zip(x_list, sample_data.y[:, 0].tolist())]
        y_target_list = [x + 100 * y for x, y in zip(y_list, sample_data.y[:, 1].tolist())]
        z_target_list = [x + 100 * y for x, y in zip(z_list, sample_data.y[:, 2].tolist())]

        dataset_loader.create_animation(sample_data, x_list, y_list, z_list, x_target_list, y_target_list,
                                        z_target_list, x_forces_list, y_forces_list, z_forces_list, supports_list, num_epochs=100)

if __name__ == "__main__":
    main()
