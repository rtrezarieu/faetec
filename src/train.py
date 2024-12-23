import os
from dotenv import load_dotenv
from comet_ml import Experiment
import torch
from tqdm import tqdm
from copy import deepcopy
import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
from .datasets.base_dataset import BaseDataset, ParallelCollater
from .modules.frame_averaging import FrameAveraging
from .faenet import FAEtec
from .datasets.data_utils import Normalizer, GraphRotate, GraphReflect
from .utils import Compose
from .gnn_utils import node_accuracy_error

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
        mode = self.config["dataset"].get("mode", "train")
        if mode == "train":
            self.load_val_loaders()
            self.load_scheduler()
            self.load_criterion()
    
    def load_logger(self):
        if not self.debug:
            if self.config['logger'] == 'comet':
                # Load environment variables from .env file 
                load_dotenv()
                api_key = os.getenv("COMET_API_KEY") # Your comet API key as a string
                workspace = os.getenv("COMET_WORKSPACE")  # Your comet workspace as a string
                if api_key is None:
                    raise ValueError("COMET_API_KEY environment variable not set")
                if workspace is None:
                    raise ValueError("COMET_WORKSPACE environment variable not set")
                self.experiment = Experiment(
                    api_key=api_key,
                    project_name=self.config['project'],
                    workspace=workspace
                )
                self.experiment.set_name(self.run_name)
                self.experiment.log_parameters(self.config)
                self.writer = self.experiment


    def load_model(self):
        model_path = self.config["dataset"].get("pretrained_model_path", None)
        if model_path:
            print(f"Loading model from {model_path}")
            self.model = FAEtec(**self.config["model"]).to(self.device)
            print(f"Loading state_dict {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            self.model = FAEtec(**self.config["model"]).to(self.device)
        
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
                self.normalizer = Normalizer(
                    means={
                        'disp': self.config['dataset']['train']['target_mean_disp'],
                        'N': self.config['dataset']['train']['target_mean_N'],
                        'M': self.config['dataset']['train']['target_mean_M']
                    },
                    stds={
                        'disp': self.config['dataset']['train']['target_std_disp'],
                        'N': self.config['dataset']['train']['target_std_N'],
                        'M': self.config['dataset']['train']['target_std_M']
                    }
                )
            else:
                self.normalizer = None

        self.parallel_collater = ParallelCollater() # To create graph batches
        self.transform = transformations_list(self.config)
        
        train_dataset = BaseDataset(self.config['dataset']['train'], transform=self.transform)
        print('Loading training dataset...')
        inconsistent_indices = [i for i, data in enumerate(train_dataset) if data.pos.shape[0] != data.y.shape[0]]
        if inconsistent_indices:
            print(f"Inconsistent data found at indices: {inconsistent_indices}")
        else:
            print("No inconsistencies found in the training dataset.")
        if self.config["dataset"].get("mode", "train") == "train":
            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config["optimizer"]['batch_size'], shuffle=False, num_workers=0, collate_fn=self.parallel_collater)
        else:
            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config["optimizer"]['eval_batch_size'], shuffle=False, num_workers=0, collate_fn=self.parallel_collater)
    
    def load_val_loaders(self):
        transform = self.transform if self.config.get('equivariance', '') != "data_augmentation" else None
        val_dataset = BaseDataset(self.config['dataset']['val']['val'], transform=transform)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config["optimizer"]['eval_batch_size'], shuffle=False, num_workers=0, collate_fn=self.parallel_collater)

    def save_model(self):
        model_save_path = self.config["dataset"].get("save_model_path", "models/saved_model.pth")
        if not os.path.exists(os.path.dirname(model_save_path)):
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_save_path)
        print(f"Model saved at {model_save_path}")

    def save_predictions(self, stored_predictions, epoch, transformations=False, base=False):
        if self.config["dataset"].get("mode", "train") == "train":
            if transformations:
                preds_save_path = self.config["dataset"].get("save_transformed_preds_path", "models/saved_transformed_predictions/")
            elif base:
                preds_save_path = self.config["dataset"].get("save_transformed_base_path", "models/saved_transformed_base/")
            else:
                preds_save_path = self.config["dataset"].get("save_preds_path", "models/saved_predictions/")
            preds_save_path = os.path.join(preds_save_path, f"epoch_{epoch}.pt")
        else:
            preds_save_path = self.config["dataset"].get("save_preds_path", "models/saved_predictions.pt")
        if not os.path.exists(os.path.dirname(preds_save_path)):
            os.makedirs(os.path.dirname(preds_save_path), exist_ok=True)
        torch.save(stored_predictions, preds_save_path)
        print(f"Predictions saved at {preds_save_path}")

    def faenet_call(self, batch, return_transformations=False):
        equivariance = self.config.get("equivariance", "")
        output_keys = ["disp", "N", "M"]
        outputs = {key: [] for key in output_keys}
        transformations = {"disp": [], "N": [], "M": []}

        if not hasattr(batch, "nnodes"):
            batch.nnodes = torch.unique(batch.batch, return_counts=True)[1]

        if equivariance != "frame_averaging":
            return self.model(batch)
        else:
            original_positions = deepcopy(batch.pos)
            original_forces = deepcopy(batch.forces)
            # The frame's positions are computed by the data loader
            # If stochastic frame averaging is used, fa_pos will only have one element
            # If the full frame is used, it will have 8 elements in 3D and 4 in 2D (OC20)
            for i in range(len(batch.fa_pos)):  # i represents the frame index (0 to 15 in 3D)
                batch.pos = batch.fa_pos[i]
                batch.forces = batch.fa_f[i]
                output = self.model(deepcopy(batch))
                
                # Displacements predictions are rotated back to be equivariant
                if output.get("disp") is not None:
                    lmbda_f = torch.repeat_interleave(batch.lmbda_f[i], batch.nnodes, dim=0).to(output["disp"].device)
                    fa_rot = torch.repeat_interleave(batch.fa_rot[i], batch.nnodes, dim=0)
                    g_disp = (
                        output["disp"]
                        .view(-1, 1, 3) # 3 for the 3D coordinates
                        .bmm(fa_rot.transpose(1, 2).to(output["disp"].device))
                    )
                    g_disp = (lmbda_f.view(-1, 1, 1) * g_disp).view(-1, 3)
                    if return_transformations:
                        transformations["disp"].append(output["disp"])
                    output["disp"] = g_disp

                if output.get("N") is not None:
                    lmbda_f = torch.repeat_interleave(batch.lmbda_f[i], batch.nnodes, dim=0).to(output["N"].device)
                    g_n = (output["N"].view(-1, 1, 18) * lmbda_f.view(-1, 1, 1)).view(-1, 18)
                    if return_transformations:
                        transformations["N"].append(output["N"])
                    output["N"] = g_n

                if output.get("M") is not None:
                    lmbda_f = torch.repeat_interleave(batch.lmbda_f[i], batch.nnodes, dim=0).to(output["M"].device)
                    g_m = (output["M"].view(-1, 1, 18) * lmbda_f.view(-1, 1, 1)).view(-1, 18)
                    if return_transformations:
                        transformations["M"].append(output["M"])
                    output["M"] = g_m

                for key in output_keys:
                    outputs[key].append(output[key])

            batch.pos = original_positions
            batch.forces = original_forces

        # Average predictions over frames
        output = {key: torch.stack(outputs[key], dim=0).mean(dim=0) for key in output_keys}
        
        if return_transformations:
            if equivariance == "frame_averaging":
                transformation = {key: torch.stack(transformations[key], dim=0).mean(dim=0) for key in output_keys}
                return output, transformation
        return output


    def train(self):
        log_interval = self.config["optimizer"].get("log_interval", 100)
        epochs = self.config["optimizer"].get("epochs", 1)

        mae = torch.nn.L1Loss()
        mse = torch.nn.MSELoss()

        run_time, n_batches = 0, 0
        self.min_val_loss = float('inf')
        for epoch in range(epochs):
            self.model.train()
            pbar = tqdm(self.train_loader)
            total_mae_disp, total_mse_disp = 0, 0
            total_mae_N, total_mse_N = 0, 0
            total_mae_M, total_mse_M = 0, 0

            for batch_idx, (batch) in enumerate(pbar):
                batch = batch[0].to(self.device)
                if not hasattr(batch, "nnodes"):
                    batch.nnodes = torch.unique(batch.batch, return_counts=True)[1]
                n_batches += len(batch.nnodes)
                self.optimizer.zero_grad()
                start_time = torch.cuda.Event(enable_timing=True)  
                output = self.faenet_call(batch)
                end_time = torch.cuda.Event(enable_timing=True)  
                start_time.record()  
                end_time.record()  
                torch.cuda.synchronize()  
                current_run_time = start_time.elapsed_time(end_time)  
                run_time += current_run_time  
                target = batch.y
                if self.normalizer:
                    target_normed = self.normalizer.norm({
                        'disp': target[:, 0:3],
                        'N': target[:, 3:21],
                        'M': target[:, 21:39]
                    })
                    target_unnormed = {
                        'disp': target[:, 0:3],
                        'N': target[:, 3:21],
                        'M': target[:, 21:39]
                    }
                    output_unnormed = self.normalizer.denorm({
                        'disp': output["disp"].reshape(-1, 3),
                        'N': output["N"].reshape(-1, 18),
                        'M': output["M"].reshape(-1, 18)
                    })
                else:
                    target_normed = {
                        'disp': target[:, 0:3],
                        'N': target[:, 3:21],
                        'M': target[:, 21:39]
                    }
                    target_unnormed = target_normed
                    output_unnormed = {
                        'disp': output["disp"].reshape(-1, 3),
                        'N': output["N"].reshape(-1, 18),
                        'M': output["M"].reshape(-1, 18)
                    }

                loss_disp = self.criterion(output["disp"].reshape(-1, 3).to(self.device), target_normed["disp"].reshape(-1, 3))
                loss_N = self.criterion(output["N"].reshape(-1, 18).to(self.device), target_normed["N"].reshape(-1, 18))
                loss_M = self.criterion(output["M"].reshape(-1, 18).to(self.device), target_normed["M"].reshape(-1, 18))
                loss = loss_disp
                # loss = loss_disp + loss_N + loss_M
                loss.backward()

                mae_loss_disp = mae(output_unnormed["disp"].to(self.device), target_unnormed["disp"]).detach()
                mae_loss_N = mae(output_unnormed["N"].to(self.device), target_unnormed["N"]).detach()
                mae_loss_M = mae(output_unnormed["M"].to(self.device), target_unnormed["M"]).detach()

                mse_loss_disp = mse(output_unnormed["disp"].to(self.device), target_unnormed["disp"]).detach()
                mse_loss_N = mse(output_unnormed["N"].to(self.device), target_unnormed["N"]).detach()
                mse_loss_M = mse(output_unnormed["M"].to(self.device), target_unnormed["M"]).detach()

                total_mae_disp += mae_loss_disp
                total_mse_disp += mse_loss_disp
                total_mae_N += mae_loss_N
                total_mse_N += mse_loss_N
                total_mae_M += mae_loss_M
                total_mse_M += mse_loss_M

                self.optimizer.step()

                metrics = {
                    "train/loss": loss.detach().item(),
                    "train/mae_disp": mae_loss_disp.item(),
                    "train/mae_N": mae_loss_N.item(),
                    "train/mae_M": mae_loss_M.item(),
                    "train/mse_disp": mse_loss_disp.item(),
                    "train/mse_N": mse_loss_N.item(),
                    "train/mse_M": mse_loss_M.item(),
                    "train/batch_run_time": current_run_time,    
                    "train/lr": self.optimizer.param_groups[0]['lr'],
                    "train/epoch": (epoch*len(self.train_loader) + batch_idx) / (len(self.train_loader))
                }

                if not self.debug:
                    if self.config['logger'] == 'comet':
                        self.writer.log_metrics(metrics)

                pbar.set_description(f'Epoch {epoch+1}/{epochs} - Loss: {loss.detach().item():.6f}')
                if self.scheduler:
                    self.scheduler.step()

            if not self.debug:
                if self.config['logger'] == 'comet':
                    self.writer.log_metrics({
                        "train/mae_disp_epoch": total_mae_disp,
                        "train/mse_disp_epoch": total_mse_disp,
                        "train/mae_N_epoch": total_mae_N,
                        "train/mse_N_epoch": total_mse_N,
                        "train/mae_M_epoch": total_mae_M,
                        "train/mse_M_epoch": total_mse_M,
                    })

            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            if not self.debug:
                if self.config['logger'] == 'comet':
                    self.writer.log_metric("systems_per_second", 1 / (run_time / n_batches))  

            self.validate(epoch)
        
        # invariance_metrics = self.measure_model_invariance(self.model)
        self.save_model()

        return self.min_val_loss
        
    def validate(self, epoch=0, splits=None):
        self.model.eval()
        mae = torch.nn.L1Loss()
        mse = torch.nn.MSELoss()
        stored_predictions = {'disp': [], 'N': [], 'M': []}
        stored_transformations = {'disp': [], 'N': [], 'M': []}
        stored_base_transformed = {'base': []}
        run_time = 0.0

        with torch.no_grad():
            if self.config["dataset"].get("mode", "train") == "train":
                val_loader = self.val_loader
            else:
                val_loader = self.train_loader
            pbar = tqdm(val_loader)
            mae_loss_disp, mse_loss_disp = 0, 0
            mae_loss_N, mse_loss_N = 0, 0
            mae_loss_M, mse_loss_M = 0, 0
            accuracy_loss_disp, accuracy_loss_N, accuracy_loss_M = 0, 0, 0
            relerror_loss_disp, relerror_loss_N, relerror_loss_M = 0, 0, 0

            for batch_idx, (batch) in enumerate(pbar):
                batch = batch[0].to(self.device)
                if self.config["dataset"].get("mode", "train") == "train": 
                    if self.config.get("equivariance", "") == "frame_averaging":
                        output, transformation = self.faenet_call(batch, return_transformations=True)
                        base_transformed = batch.fa_pos[0]
                        if batch_idx == 0:
                            rotation_matrix = batch.fa_rot[0][0]
                            print(f"Rotation matrix: {rotation_matrix}")
                    else:
                        output = self.faenet_call(batch)
                        transformation = output  # No transformation for predictions
                        base_transformed = batch.pos # No transformation for the base model
                        target_transformed = batch.y 
                else:
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)
                    start_time.record()   
                    output = self.faenet_call(batch)
                    end_time.record()  
                    torch.cuda.synchronize()  
                    current_run_time = start_time.elapsed_time(end_time)  
                    run_time += current_run_time
                    print(f"Elapsed time for predictions: {current_run_time} ms")
                target = batch.y

                if self.normalizer:
                    output_unnormed = self.normalizer.denorm({
                        'disp': output["disp"].reshape(-1, 3),
                        'N': output["N"].reshape(-1, 18),
                        'M': output["M"].reshape(-1, 18)
                    })
                    transformation_unnormed = self.normalizer.denorm({
                        'disp': transformation["disp"].reshape(-1, 3),
                        'N': transformation["N"].reshape(-1, 18),
                        'M': transformation["M"].reshape(-1, 18)
                    })
                else:
                    output_unnormed = {
                        'disp': output["disp"].reshape(-1, 3),
                        'N': output["N"].reshape(-1, 18),
                        'M': output["M"].reshape(-1, 18)
                    }
                    transformation_unnormed = {
                        'disp': transformation["disp"].reshape(-1, 3),
                        'N': transformation["N"].reshape(-1, 18),
                        'M': transformation["M"].reshape(-1, 18)
                    }
                    
                target_unnormed = {
                    'disp': target[:, 0:3],
                    'N': target[:, 3:21],
                    'M': target[:, 21:39]
                }

                stored_predictions['disp'].append(output_unnormed['disp'].cpu())
                stored_predictions['N'].append(output_unnormed['N'].cpu())
                stored_predictions['M'].append(output_unnormed['M'].cpu())

                stored_transformations['disp'].append(transformation_unnormed['disp'].cpu())
                stored_transformations['N'].append(transformation_unnormed['N'].cpu())
                stored_transformations['M'].append(transformation_unnormed['M'].cpu())

                stored_base_transformed['base'].append(base_transformed[:, 0:3].cpu())

                mae_loss_disp_batch = mae(output_unnormed["disp"].to(self.device), target_unnormed["disp"]).detach()  
                mse_loss_disp_batch = mse(output_unnormed["disp"].to(self.device), target_unnormed["disp"]).detach()

                mae_loss_N_batch = mae(output_unnormed["N"].to(self.device), target_unnormed["N"]).detach()
                mse_loss_N_batch = mse(output_unnormed["N"].to(self.device), target_unnormed["N"]).detach()

                mae_loss_M_batch = mae(output_unnormed["M"].to(self.device), target_unnormed["M"]).detach()
                mse_loss_M_batch = mse(output_unnormed["M"].to(self.device), target_unnormed["M"]).detach()

                mae_loss_disp += mae_loss_disp_batch
                mse_loss_disp += mse_loss_disp_batch

                mae_loss_N += mae_loss_N_batch
                mse_loss_N += mse_loss_N_batch

                mae_loss_M += mae_loss_M_batch
                mse_loss_M += mse_loss_M_batch

                accuracy_loss_disp_batch, relerror_loss_disp_batch, num_disp = node_accuracy_error(output_unnormed["disp"].to(self.device), target_unnormed["disp"], accuracy_threshold=0.1*self.config['dataset']['train']['target_std_disp'], disp=True)
                accuracy_loss_N_batch, relerror_loss_N_batch, num_N = node_accuracy_error(output_unnormed["N"].to(self.device), target_unnormed["N"], accuracy_threshold=1e-15*self.config['dataset']['train']['target_std_N'], disp=False)
                accuracy_loss_M_batch, relerror_loss_M_batch, num_M = node_accuracy_error(output_unnormed["M"].to(self.device), target_unnormed["M"], accuracy_threshold=1e-15*self.config['dataset']['train']['target_std_M'], disp=False)

                accuracy_loss_disp += accuracy_loss_disp_batch / num_disp
                accuracy_loss_N += accuracy_loss_N_batch / num_N
                accuracy_loss_M += accuracy_loss_M_batch / num_M
                
                relerror_loss_disp += relerror_loss_disp_batch / num_disp
                relerror_loss_N += relerror_loss_N_batch / num_N
                relerror_loss_M += relerror_loss_M_batch / num_M

                pbar.set_description(
                    f'Val - Epoch {epoch+1} - '
                    f'MAE Disp: {mae_loss_disp.item()/(batch_idx+1):.6f}, '
                    f'N: {mae_loss_N.item()/(batch_idx+1):.6f}, M: {mae_loss_M.item()/(batch_idx+1):.6f}, '
                    f'Acc Disp: {accuracy_loss_disp.item() / (batch_idx+1):.6f}, '
                    f'Acc N: {accuracy_loss_N.item() / (batch_idx+1):.6f}, '
                    f'Acc M: {accuracy_loss_M.item() / (batch_idx+1):.6f}, '
                    f'Rel Disp: {relerror_loss_disp.item() / (batch_idx+1):.6f}, '
                    f'Rel N: {relerror_loss_N.item() / (batch_idx+1):.6f}, '
                    f'Rel M: {relerror_loss_M.item() / (batch_idx+1):.6f}'
                )

            stored_predictions['disp'] = torch.cat(stored_predictions['disp'], dim=0)
            stored_predictions['N'] = torch.cat(stored_predictions['N'], dim=0)
            stored_predictions['M'] = torch.cat(stored_predictions['M'], dim=0)

            stored_transformations['disp'] = torch.cat(stored_transformations['disp'], dim=0)
            stored_transformations['N'] = torch.cat(stored_transformations['N'], dim=0)
            stored_transformations['M'] = torch.cat(stored_transformations['M'], dim=0)

            stored_base_transformed['base'] = torch.cat(stored_base_transformed['base'], dim=0)

            if self.config['save_predictions']:
                self.save_predictions(stored_predictions, epoch)
                self.save_predictions(stored_transformations, epoch, transformations=True)
                self.save_predictions(stored_base_transformed, epoch, base=True)

            # Calculate average losses over the entire validation set - len(val_loader) is the number of batches
            total_mae_disp = mae_loss_disp.item() / len(val_loader)
            total_mse_disp = mse_loss_disp.item() / len(val_loader)

            total_mae_N = mae_loss_N.item() / len(val_loader)
            total_mse_N = mse_loss_N.item() / len(val_loader)

            total_mae_M = mae_loss_M.item() / len(val_loader)
            total_mse_M = mse_loss_M.item() / len(val_loader)

            total_accuracy_loss_disp = accuracy_loss_disp / len(val_loader)
            total_accuracy_loss_N = accuracy_loss_N / len(val_loader)
            total_accuracy_loss_M = accuracy_loss_M / len(val_loader)

            total_relerror_loss_disp = relerror_loss_disp / len(val_loader)
            total_relerror_loss_N = relerror_loss_N / len(val_loader)
            total_relerror_loss_M = relerror_loss_M / len(val_loader)

            if self.config["dataset"].get("mode", "train") == "train":
                if total_mae_disp < self.min_val_loss:
                    self.min_val_loss = total_mae_disp

            if not self.debug:
                metrics = {
                    f"val/mae_disp": total_mae_disp,
                    f"val/mse_disp": total_mse_disp,
                    f"val/mae_N": total_mae_N,
                    f"val/mse_N": total_mse_N,
                    f"val/mae_M": total_mae_M,
                    f"val/mse_M": total_mse_M,
                    f"val/accuracy_loss_disp": total_accuracy_loss_disp,
                    f"val/accuracy_loss_N": total_accuracy_loss_N,
                    f"val/accuracy_loss_M": total_accuracy_loss_M,
                    f"val/relerror_loss_disp": total_relerror_loss_disp,
                    f"val/relerror_loss_N": total_relerror_loss_N,
                    f"val/relerror_loss_M": total_relerror_loss_M,
                }
                if self.config['logger'] == 'comet':
                    self.writer.log_metrics(metrics)
        

    # FAEnet implementation - untouched
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

            n_batches += batch[0].pos.shape[0]

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

        metrics[f"energy_delta_rotated_2d"] = energy_delta_rotated_2d / n_batches
        metrics[f"energy_delta_reflected"] =  energy_delta_reflected / n_batches
        metrics[f"energy_delta_rotated_3d"] =  energy_delta_rotated_3d / n_batches
        if not self.debug:
            if self.config['logger'] == 'comet':
                self.writer.log_metrics(metrics)
        pbar.close()
        print("\nInvariance results:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

        return metrics