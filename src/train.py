from comet_ml import Experiment
import torch
import wandb
from tqdm import tqdm
import numpy as np
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
            elif self.config['logger'] == 'comet':
                self.experiment = Experiment(
                    api_key="4PNGEKdzZGpTM83l7pBrnYgTo",
                    project_name=self.config['project'],
                    workspace="rtrezarieu"
                )
                self.experiment.set_name(self.run_name)
                self.experiment.log_parameters(self.config)
                self.writer = self.experiment
    
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
                # self.normalizer = Normalizer(means=self.config['dataset']['train']['target_mean'], stds=self.config['dataset']['train']['target_std'])
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
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config["optimizer"]['batch_size'], shuffle=True, num_workers=0, collate_fn=self.parallel_collater)
    
    # Valide explicitement sur un set tourné
    # Valide sur une structure tournée plusieurs fois, si config['equivariance'] != 'data_augmentation'
    def load_val_loaders(self):
        self.val_loaders = []
        for split in self.config['dataset']['val']: #### pas de split ici??
            transform = self.transform if self.config.get('equivariance', '') != "data_augmentation" else None
            val_dataset = BaseDataset(self.config['dataset']['val'][split], transform=transform)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config["optimizer"]['eval_batch_size'], shuffle=False, num_workers=0, collate_fn=self.parallel_collater)
            self.val_loaders.append(val_loader)
    

    # Each element of the batch corresponds to one node of one structure
    def faenet_call(self, batch):
        equivariance = self.config.get("equivariance", "")
        output_keys = ["disp", "N", "M"]  # output contains the predictions for a single frame
        outputs = {key: [] for key in output_keys}  # outputs collects predictions from all frames
        # disp_all, m_all, n_all = [], [], []

        # if isinstance(batch, list):
        #     batch = batch[0]
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
                    lmbda_f = torch.repeat_interleave(batch.lmbda_f[i], batch.nnodes, dim=0)
                    fa_rot = torch.repeat_interleave(batch.fa_rot[i], batch.nnodes, dim=0)
                    g_disp = (
                        output["disp"]
                        .view(-1, 1, 3) # 3 for the 3D coordinates
                        .bmm(fa_rot.transpose(1, 2).to(output["disp"].device))
                    )
                    g_disp = (lmbda_f.view(-1, 1, 1) * g_disp).view(-1, 3)
                    output["disp"] = g_disp
                    # disp_all.append(g_disp)

                if output.get("M") is not None:
                    lmbda_f = torch.repeat_interleave(batch.lmbda_f[i], batch.nnodes, dim=0)
                    g_m = (output["M"].view(-1, 1, 18) * lmbda_f.view(-1, 1, 1)).view(-1, 18)
                    output["M"] = g_m
                    # m_all.append(g_m)

                if output.get("N") is not None:
                    lmbda_f = torch.repeat_interleave(batch.lmbda_f[i], batch.nnodes, dim=0)
                    g_n = (output["N"].view(-1, 1, 18) * lmbda_f.view(-1, 1, 1)).view(-1, 18)
                    output["N"] = g_n
                    # n_all.append(g_n)

                for key in output_keys:
                    outputs[key].append(output[key])

            batch.pos = original_positions
            batch.forces = original_forces

        # Average predictions over frames
        output = {key: torch.stack(outputs[key], dim=0).mean(dim=0) for key in output_keys}

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
            total_mae_disp, total_mse_disp = 0, 0
            total_mae_N, total_mse_N = 0, 0
            total_mae_M, total_mse_M = 0, 0

            n_batches_epoch = 0
            for batch_idx, (batch) in enumerate(pbar):
                # n_batches += len(batch[0].natoms)
                n_batches += batch[0].pos.shape[0]      ######### erreur????   batch.nnodes = batch.natoms
                n_batches_epoch += batch[0].pos.shape[0]  #################################  erreur????
                batch = batch[0].to(self.device)
                self.optimizer.zero_grad()
                # start_time = 0      #############################  à commenter pour exe locale
                # start_time = torch.cuda.Event(enable_timing=True)  #############################  à commenter pour exe locale
                output = self.faenet_call(batch)
                # end_time = 0  #############################  à commenter pour exe locale
                # end_time = torch.cuda.Event(enable_timing=True)  #############################  à commenter pour exe locale
                # start_time.record()  #############################  à commenter pour exe locale
                # end_time.record()  #############################  à commenter pour exe locale
                # torch.cuda.synchronize()  #############################  à commenter pour exe locale
                # current_run_time = start_time.elapsed_time(end_time)  #############################  à commenter pour exe locale
                # current_run_time = 0  #############################  à commenter pour exe locale
                # run_time += current_run_time  #############################  à commenter pour exe locale
                target = batch.y
                if self.normalizer:
                    target_normed = self.normalizer.norm({
                        'disp': target[:, 0:3],
                        'N': target[:, 3:21],
                        'M': target[:, 21:39]
                    })
                    # output_unnormed = self.normalizer.denorm(output["energy"].reshape(-1))
                    output_unnormed = self.normalizer.denorm({
                        'disp': output["disp"].reshape(-1, 3),
                        'N': output["N"].reshape(-1, 18),
                        'M': output["M"].reshape(-1, 18)
                    })
                else:
                    # target_normed = target
                    target_normed = {
                        'disp': target[:, 0:3],
                        'N': target[:, 3:21],
                        'M': target[:, 21:39]
                    }
                    # output_unnormed = output["energy"].reshape(-1)
                    output_unnormed = {
                        'disp': output["disp"].reshape(-1, 3),
                        'N': output["N"].reshape(-1, 18),
                        'M': output["M"].reshape(-1, 18)
                    }

                loss_disp = self.criterion(output["disp"].reshape(-1, 3), target_normed["disp"].reshape(-1, 3))
                loss_N = self.criterion(output["N"].reshape(-1, 18), target_normed["N"].reshape(-1, 18))
                loss_M = self.criterion(output["M"].reshape(-1, 18), target_normed["M"].reshape(-1, 18))
                # Losses combination (weights can be adjusted)
                loss = loss_disp + loss_N + loss_M
                loss.backward()

                mae_loss_disp = mae(output_unnormed["disp"], target_normed["disp"]).detach()
                mae_loss_N = mae(output_unnormed["N"], target_normed["N"]).detach()
                mae_loss_M = mae(output_unnormed["M"], target_normed["M"]).detach()

                mse_loss_disp = mse(output_unnormed["disp"], target_normed["disp"]).detach()
                mse_loss_N = mse(output_unnormed["N"], target_normed["N"]).detach()
                mse_loss_M = mse(output_unnormed["M"], target_normed["M"]).detach()

                total_mae_disp += mae_loss_disp
                total_mse_disp += mse_loss_disp
                total_mae_N += mae_loss_N
                total_mse_N += mse_loss_N
                total_mae_M += mae_loss_M
                total_mse_M += mse_loss_M

                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

                metrics = {
                    "train/loss": loss.detach().item(),
                    "train/mae_disp": mae_loss_disp.item(),
                    "train/mae_N": mae_loss_N.item(),
                    "train/mae_M": mae_loss_M.item(),
                    "train/mse_disp": mse_loss_disp.item(),
                    "train/mse_N": mse_loss_N.item(),
                    "train/mse_M": mse_loss_M.item(),
                    # "train/batch_run_time": current_run_time,    #############################  à commenter pour exe locale
                    "train/lr": self.optimizer.param_groups[0]['lr'],
                    "train/epoch": (epoch*len(self.train_loader) + batch_idx) / (len(self.train_loader))
                }

                if not self.debug:
                    if self.config['logger'] == 'wandb':
                        self.writer.log(metrics)
                    elif self.config['logger'] == 'comet':
                        self.writer.log_metrics(metrics)

                pbar.set_description(f'Epoch {epoch+1}/{epochs} - Loss: {loss.detach().item():.6f}')
                if self.scheduler:
                    self.scheduler.step()

            if not self.debug:
                if self.config['logger'] == 'wandb':
                    self.writer.log({
                        "train/mae_disp_epoch": total_mae_disp,
                        "train/mse_disp_epoch": total_mse_disp,
                        "train/mae_N_epoch": total_mae_N,
                        "train/mse_N_epoch": total_mse_N,
                        "train/mae_M_epoch": total_mae_M,
                        "train/mse_M_epoch": total_mse_M,
                    })
                elif self.config['logger'] == 'comet':
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
                if self.config['logger'] == 'wandb':
                    # self.writer.log({"systems_per_second": 1 / (run_time / n_batches)})  #############################  à commenter pour exe locale
                    pass
                elif self.config['logger'] == 'comet':
                    # self.writer.log_metric("systems_per_second", 1 / (run_time / n_batches))  #############################  à commenter pour exe locale
                    pass

            if epoch != epochs-1:
                self.validate(epoch, splits=[0]) # Validate on the first split (val_id)

        self.validate(epoch) # Validate on all splits
        # invariance_metrics = self.measure_model_invariance(self.model)
        
        
    def validate(self, epoch, splits=None):    ##################################### comprendre et enlever les splits?
        self.model.eval()
        mae = torch.nn.L1Loss()
        mse = torch.nn.MSELoss()
        for i, val_loader in enumerate(self.val_loaders):
            if splits and i not in splits:
                continue
            split = list(self.config['dataset']['val'].keys())[i]
            pbar = tqdm(val_loader)
            mae_loss_disp, mse_loss_disp = 0, 0
            mae_loss_N, mse_loss_N = 0, 0
            mae_loss_M, mse_loss_M = 0, 0
            n_batches = 0

            for batch_idx, (batch) in enumerate(pbar):
                n_batches += batch[0].pos.shape[0]   ################ peut-être à changer         n_batches += len(batch[0].natoms)
                batch = batch[0].to(self.device)
                output = self.faenet_call(batch)
                target = batch.y

                if self.normalizer:
                    # output_unnormed = self.normalizer.denorm(output["energy"].reshape(-1))
                    output_unnormed = self.normalizer.denorm({
                        'disp': output["disp"].reshape(-1, 3),
                        'N': output["N"].reshape(-1, 18),
                        'M': output["M"].reshape(-1, 18)
                    })
                else:
                    # output_unnormed = output["energy"].reshape(-1)
                    output_unnormed = {
                        'disp': output["disp"].reshape(-1, 3),
                        'N': output["N"].reshape(-1, 18),
                        'M': output["M"].reshape(-1, 18)
                    }
                    
                target_unnormed = {
                    'disp': target[:, 0:3],
                    'N': target[:, 3:21],
                    'M': target[:, 21:39]
                }

                # Compute MAE and MSE for each output (disp, N, M)
                mae_loss_disp_batch = mae(output_unnormed["disp"], target_unnormed["disp"]).detach()
                mse_loss_disp_batch = mse(output_unnormed["disp"], target_unnormed["disp"]).detach()

                mae_loss_N_batch = mae(output_unnormed["N"], target_unnormed["N"]).detach()
                mse_loss_N_batch = mse(output_unnormed["N"], target_unnormed["N"]).detach()

                mae_loss_M_batch = mae(output_unnormed["M"], target_unnormed["M"]).detach()
                mse_loss_M_batch = mse(output_unnormed["M"], target_unnormed["M"]).detach()

                # Accumulate the losses
                mae_loss_disp += mae_loss_disp_batch
                mse_loss_disp += mse_loss_disp_batch

                mae_loss_N += mae_loss_N_batch
                mse_loss_N += mse_loss_N_batch

                mae_loss_M += mae_loss_M_batch
                mse_loss_M += mse_loss_M_batch


                pbar.set_description(
                    f'Val {i} - Epoch {epoch+1} - MAE Disp: {mae_loss_disp.item()/(batch_idx+1):.6f}, '
                    f'N: {mae_loss_N.item()/(batch_idx+1):.6f}, M: {mae_loss_M.item()/(batch_idx+1):.6f}'
                )
                # pbar.set_description(f'Val {i} - Epoch {epoch+1} - MAE: {mae_loss.item()/(batch_idx+1):.6f}')

            # Calculate average losses over the entire validation set
            total_mae_disp = mae_loss_disp.item() / len(val_loader)
            total_mse_disp = mse_loss_disp.item() / len(val_loader)

            total_mae_N = mae_loss_N.item() / len(val_loader)
            total_mse_N = mse_loss_N.item() / len(val_loader)

            total_mae_M = mae_loss_M.item() / len(val_loader)
            total_mse_M = mse_loss_M.item() / len(val_loader)

            if not self.debug:
                metrics = {
                    f"{split}/mae_disp": total_mae_disp,
                    f"{split}/mse_disp": total_mse_disp,
                    f"{split}/mae_N": total_mae_N,
                    f"{split}/mse_N": total_mse_N,
                    f"{split}/mae_M": total_mae_M,
                    f"{split}/mse_M": total_mse_M,
                }
                if self.config['logger'] == 'wandb':
                    self.writer.log(metrics)
                elif self.config['logger'] == 'comet':
                    self.writer.log_metrics(metrics)


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

        metrics[f"{split}/energy_delta_rotated_2d"] = energy_delta_rotated_2d / n_batches
        metrics[f"{split}/energy_delta_reflected"] =  energy_delta_reflected / n_batches
        metrics[f"{split}/energy_delta_rotated_3d"] =  energy_delta_rotated_3d / n_batches
        if not self.debug:
            if self.config['logger'] == 'wandb':
                self.writer.log(metrics)
            elif self.config['logger'] == 'comet':
                self.writer.log_metrics(metrics)
        pbar.close()
        print("\nInvariance results:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

        return metrics