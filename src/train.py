import torch
import wandb
from tqdm import tqdm
from copy import deepcopy
import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
from .datasets.base_dataset import BaseDataset, ParallelCollater
from .modules.frame_averaging import FrameAveraging
from .faenet import FAENet
from .datasets.data_utils import Normalizer, GraphRotate, GraphReflect
from .utils import Compose

def transformations_list(config):
    transform_list = []
    if config.get('equivariance', "") != "":
        transform_list.append(FrameAveraging(config['equivariance'], config['fa_type'], config['dataset'].get('oc20', True)))
    if len(transform_list) > 0:
        return Compose(transform_list)
    else:
        return None


class Trainer():
    def __init__(self, config, device="cpu", debug=False):
        self.config = config
        self.debug = debug
        self.device = device
        self.run_name = f"{self.config['run_name']}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self.device
        self.load()

    def load(self):
        self.load_model()
        self.load_logger()
        self.load_optimizer()
        self.load_train_loader()
        self.load_val_loaders()
        self.load_scheduler()
        self.load_criterion()
    
    def load_logger(self):
        if not self.debug:
            if self.config['logger'] == 'wandb':
                wandb.init(project=self.config['project'], name=self.run_name)
                wandb.config.update(self.config)
                self.writer = wandb
    
    def load_model(self):
        self.model = FAENet(**self.config["model"]).to(self.device)
    
    def load_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['optimizer'].get('lr_initial', 1e-4))
    
    def load_scheduler(self):
        if self.config['optimizer'].get('scheduler', None) == 'CosineAnnealingLR':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config['optimizer'].get('epochs', 1)*len(self.train_loader))
        else:
            self.scheduler = None
    
    def load_criterion(self, reduction='mean'):
        if self.config['model'].get('loss', 'mse') == 'mse':
            self.criterion = torch.nn.MSELoss(reduction=reduction)
        elif self.config['model'].get('loss', 'mse') == 'mae':
            self.criterion = torch.nn.L1Loss(reduction=reduction)
    
    def load_train_loader(self):
        if self.config['dataset']['train'].get("normalize_labels", False):
            if self.config['dataset']['train']['normalize_labels']:
                self.normalizer = Normalizer( mean=self.config['dataset']['train']['target_mean'], std=self.config['dataset']['train']['target_std'])
            else:
                self.normalizer = None

        self.parallel_collater = ParallelCollater() # To create graph batches
        self.transform = transformations_list(self.config)
        train_dataset = BaseDataset(self.config['dataset']['train'], transform=self.transform)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config["optimizer"]['batch_size'], shuffle=True, num_workers=0, collate_fn=self.parallel_collater)
    
    def load_val_loaders(self):
        self.val_loaders = []
        for split in self.config['dataset']['val']:
            transform = self.transform if self.config.get('equivariance', '') != "data_augmentation" else None
            val_dataset = BaseDataset(self.config['dataset']['val'][split], transform=transform)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config["optimizer"]['eval_batch_size'], shuffle=False, num_workers=0, collate_fn=self.parallel_collater)
            self.val_loaders.append(val_loader)
    
    def faenet_call(self, batch):
        equivariance = self.config.get("equivariance", "")
        outputs = []
        if equivariance != "frame_averaging":
            return self.model(batch)
        else:
            original_positions = deepcopy(batch.pos)
            if hasattr(batch, "cell"):
                original_cell = deepcopy(batch.fa_cell)
            # The frame's positions are computed by the data loader
            # If stochastic frame averaging is used, fa_pos will only have one element
            # If the full frame is used, it will have 8 elements in 3D and 4 in 2D (OC20)
            for i in range(len(batch.fa_pos)):
                batch.pos = batch.fa_pos[i]
                if hasattr(batch, "cell"):
                    batch.cell = batch.fa_cell[i]
                output = self.model(deepcopy(batch))
                outputs.append(output["energy"])
            batch.pos = original_positions
            if hasattr(batch, "cell"):
                batch.cell = original_cell
        energy_prediction = torch.stack(outputs, dim=0).mean(dim=0)
        output["energy"] = energy_prediction
        return output

    def train(self):
        log_interval = self.config["optimizer"].get("log_interval", 100)
        epochs = self.config["optimizer"].get("epochs", 1)

        mae = torch.nn.L1Loss()
        mse = torch.nn.MSELoss()

        run_time, n_batches = 0, 0
        for epoch in range(epochs):
            self.model.train()
            pbar = tqdm(self.train_loader)
            mae_loss, mse_loss = 0, 0
            n_batches_epoch = 0
            for batch_idx, (batch) in enumerate(pbar):
                n_batches += len(batch[0].natoms)
                n_batches_epoch += len(batch[0].natoms)
                batch = batch[0].to(self.device)
                self.optimizer.zero_grad()
                start_time = torch.cuda.Event(enable_timing=True)
                output = self.faenet_call(batch)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()
                end_time.record()
                torch.cuda.synchronize()
                current_run_time = start_time.elapsed_time(end_time)
                run_time += current_run_time
                target = batch.y_relaxed
                if self.normalizer:
                    target_normed = self.normalizer.norm(target)
                    output_unnormed = self.normalizer.denorm(output["energy"].reshape(-1))
                else:
                    target_normed = target
                    output_unnormed = output["energy"].reshape(-1)
                loss = self.criterion(output["energy"].reshape(-1), target_normed.reshape(-1))
                loss.backward()
                mae_loss_batch = mae(output_unnormed, target).detach()
                mse_loss_batch = mse(output_unnormed, target).detach()
                mae_loss += mae_loss_batch
                mse_loss += mse_loss_batch
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()
                metrics = {
                    "train/loss": loss.detach().item(),
                    "train/mae": mae_loss_batch.item(),
                    "train/mse": mse_loss_batch.item(),
                    "train/batch_run_time": current_run_time,
                    "train/lr": self.optimizer.param_groups[0]['lr'],
                    "train/epoch": (epoch*len(self.train_loader) + batch_idx) / (len(self.train_loader))
                }
                if not self.debug:
                    self.writer.log(metrics)
                pbar.set_description(f'Epoch {epoch+1}/{epochs} - Loss: {loss.detach().item():.6f}')
                if self.scheduler:
                    self.scheduler.step()
            if not self.debug:
                self.writer.log({
                    "train/mae_epoch": mae_loss.item()/len(self.train_loader),
                    "train/mse_epoch": mse_loss.item()/len(self.train_loader),
                })
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            if not self.debug:
                self.writer.log({"systems_per_second": 1/(run_time / n_batches)})
            if epoch != epochs-1:
                self.validate(epoch, splits=[0]) # Validate on the first split (val_id)
        self.validate(epoch) # Validate on all splits
        invariance_metrics = self.measure_model_invariance(self.model)
        
        
    def validate(self, epoch, splits=None):
        self.model.eval()
        mae = torch.nn.L1Loss()
        mse = torch.nn.MSELoss()
        for i, val_loader in enumerate(self.val_loaders):
            if splits and i not in splits:
                continue
            split = list(self.config['dataset']['val'].keys())[i]
            pbar = tqdm(val_loader)
            total_loss = 0
            mae_loss, mse_loss = 0, 0
            n_batches = 0
            for batch_idx, (batch) in enumerate(pbar):
                n_batches += len(batch[0].natoms)
                batch = batch[0].to(self.device)
                output = self.faenet_call(batch)
                target = batch.y_relaxed
                if self.normalizer:
                    output_unnormed = self.normalizer.denorm(output["energy"].reshape(-1))
                else:
                    output_unnormed = output["energy"].reshape(-1)
                mae_loss_batch = mae(output_unnormed, target).detach()
                mse_loss_batch = mse(output_unnormed, target).detach()
                mae_loss += mae_loss_batch
                mse_loss += mse_loss_batch
                pbar.set_description(f'Val {i} - Epoch {epoch+1} - MAE: {mae_loss.item()/(batch_idx+1):.6f}')
            total_loss /= len(val_loader)
            if not self.debug:
                self.writer.log({
                    f"{split}/mae": mae_loss.item() / len(val_loader),
                    f"{split}/mse": mse_loss.item() / len(val_loader),
                })

    def measure_model_invariance(self, model):
        model.eval()
        metrics = {}

        rotater_2d = GraphRotate(-180, 180, [2])
        rotater_3d = GraphRotate(-180, 180, [0, 1, 2])
        reflector = GraphReflect()

        energy_delta_rotated_2d, energy_delta_reflected, energy_delta_rotated_3d = 0, 0, 0
        n_batches = 0
        # Testing no val_id only
        new_val_loader = torch.utils.data.DataLoader(self.val_loaders[0].dataset, batch_size=max(self.config["optimizer"]['eval_batch_size'], 1), shuffle=False, num_workers=0, collate_fn=self.parallel_collater)
        split = list(self.config['dataset']['val'].keys())[0]
        pbar = tqdm(new_val_loader)
        for j, batch in enumerate(pbar): 
            batch = batch[0]
            rotated_graph, _, _ = rotater_2d(batch)
            rotated_graph_3d, _, _ = rotater_3d(batch)
            reflected_graph, _, _ = reflector(batch)

            n_batches += len(batch[0].natoms)

            preds_original = self.faenet_call(batch.to(self.device))
            del batch
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            preds_rotated = self.faenet_call(rotated_graph.to(self.device))
            energy_delta_rotated_2d += torch.abs(preds_original["energy"] - preds_rotated["energy"]).sum().detach().item()
            del rotated_graph
            del preds_rotated
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            preds_rotated_3d = self.faenet_call(rotated_graph_3d.to(self.device))
            energy_delta_rotated_3d += torch.abs(preds_original["energy"] - preds_rotated_3d["energy"]).sum().detach().item()
            del rotated_graph_3d
            del preds_rotated_3d
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            preds_reflected = self.faenet_call(reflected_graph.to(self.device))
            energy_delta_reflected += torch.abs(preds_original["energy"] - preds_reflected["energy"]).sum().detach().item()
            del reflected_graph
            del preds_reflected
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            pbar.set_description(f'Measuring invariance - Split {split} - Batch {j} - Energy Delta Rotated 2D: {energy_delta_rotated_2d/n_batches:.6f} - Energy Delta Reflected: {energy_delta_reflected/n_batches:.6f} - Energy Delta Rotated 3D: {energy_delta_rotated_3d/n_batches:.6f}')

        metrics[f"{split}/energy_delta_rotated_2d"] = energy_delta_rotated_2d / n_batches
        metrics[f"{split}/energy_delta_reflected"] =  energy_delta_reflected / n_batches
        metrics[f"{split}/energy_delta_rotated_3d"] =  energy_delta_rotated_3d / n_batches
        if not self.debug:
            self.writer.log(metrics)
        pbar.close()
        print("\nInvariance results:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

        return metrics