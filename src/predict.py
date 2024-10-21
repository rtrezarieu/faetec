from comet_ml import Experiment
import torch
from tqdm import tqdm
from copy import deepcopy
import datetime
from .datasets.base_dataset import BaseDataset, ParallelCollater
from .modules.frame_averaging import FrameAveraging
from .faenet import FAENet
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

class Predictor():
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
        self.load_pred_loader()
    
    def load_logger(self):
        if not self.debug:
            if self.config['logger'] == 'comet':
                self.experiment = Experiment(
                    api_key="4PNGEKdzZGpTM83l7pBrnYgTo",
                    project_name=self.config['project'],
                    workspace="rtrezarieu"
                )
                self.experiment.set_name(self.run_name)
                self.experiment.log_parameters(self.config)
                self.writer = self.experiment
    
    # def load_model(self):
    #     self.model = FAENet(**self.config["model"]).to(self.device)   ######### à remplacer par load model sauvegardé

    def load_model(self):
        model_path = self.config["dataset"].get("pretrained_model_path", None)
        if model_path:
            self.model = FAENet(**self.config["model"]).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            # self.model = FAENet(**self.config["model"]).to(self.device)
            raise ValueError("No model path provided.")
            
    def load_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['optimizer'].get('lr_initial', 1e-4))
    
    def load_pred_loader(self):
        if self.config['dataset']['pred'].get("normalize_labels", False):
            if self.config['dataset']['pred']['normalize_labels']:
                self.normalizer = Normalizer(
                    means={
                        'disp': self.config['dataset']['pred']['target_mean_disp'],
                        'N': self.config['dataset']['pred']['target_mean_N'],
                        'M': self.config['dataset']['pred']['target_mean_M']
                    },
                    stds={
                        'disp': self.config['dataset']['pred']['target_std_disp'],
                        'N': self.config['dataset']['pred']['target_std_N'],
                        'M': self.config['dataset']['pred']['target_std_M']
                    }
                )
            else:
                self.normalizer = None

        self.parallel_collater = ParallelCollater() # To create graph batches
        self.transform = transformations_list(self.config)
        
        pred_dataset = BaseDataset(self.config['dataset']['pred'], transform=self.transform)
        print('Loading prediction dataset...')
        inconsistent_indices = [i for i, data in enumerate(pred_dataset) if data.pos.shape[0] != data.y.shape[0]]
        if inconsistent_indices:
            print(f"Inconsistent data found at indices: {inconsistent_indices}")
        else:
            print("No inconsistencies found in the prediction dataset.")
        self.pred_loader = torch.utils.data.DataLoader(pred_dataset, batch_size=self.config["optimizer"]['eval_batch_size'], shuffle=False, num_workers=0, collate_fn=self.parallel_collater)


    def faenet_call(self, batch):
        equivariance = self.config.get("equivariance", "")
        output_keys = ["disp", "N", "M"]
        outputs = {key: [] for key in output_keys}
        # disp_all, m_all, n_all = [], [], []

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
                    output["disp"] = g_disp
                    # disp_all.append(g_disp)

                if output.get("N") is not None:
                    lmbda_f = torch.repeat_interleave(batch.lmbda_f[i], batch.nnodes, dim=0).to(output["N"].device)
                    g_n = (output["N"].view(-1, 1, 18) * lmbda_f.view(-1, 1, 1)).view(-1, 18)
                    output["N"] = g_n
                    # n_all.append(g_n)

                if output.get("M") is not None:
                    lmbda_f = torch.repeat_interleave(batch.lmbda_f[i], batch.nnodes, dim=0).to(output["M"].device)
                    g_m = (output["M"].view(-1, 1, 18) * lmbda_f.view(-1, 1, 1)).view(-1, 18)
                    output["M"] = g_m
                    # m_all.append(g_m)

                for key in output_keys:
                    outputs[key].append(output[key])

            batch.pos = original_positions
            batch.forces = original_forces

        # Average predictions over frames
        output = {key: torch.stack(outputs[key], dim=0).mean(dim=0) for key in output_keys}

        return output
        
        
    def validate(self):
        self.model.eval()
        mae = torch.nn.L1Loss()
        mse = torch.nn.MSELoss()
        run_time = 0

        with torch.no_grad():
            pred_loader = self.pred_loader
            pbar = tqdm(pred_loader)
            mae_loss_disp, mse_loss_disp = 0, 0
            mae_loss_N, mse_loss_N = 0, 0
            mae_loss_M, mse_loss_M = 0, 0
            accuracy_loss_disp, accuracy_loss_N, accuracy_loss_M = 0, 0, 0
            relerror_loss_disp, relerror_loss_N, relerror_loss_M = 0, 0, 0

            for batch_idx, (batch) in enumerate(pbar):
                batch = batch[0].to(self.device)
                
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
                    output_unnormed = self.normalizer.denorm({
                        'disp': output["disp"].reshape(-1, 3),
                        'N': output["N"].reshape(-1, 18),
                        'M': output["M"].reshape(-1, 18)
                    })
                else:
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

                accuracy_loss_disp_batch, relerror_loss_disp_batch, num_disp = node_accuracy_error(output_unnormed["disp"].to(self.device), target_unnormed["disp"], accuracy_threshold=0.1*self.config['dataset']['pred']['target_std_disp'])
                accuracy_loss_N_batch, relerror_loss_N_batch, num_N = node_accuracy_error(output_unnormed["N"].to(self.device), target_unnormed["N"], accuracy_threshold=0.1*self.config['dataset']['pred']['target_std_N'])
                accuracy_loss_M_batch, relerror_loss_M_batch, num_M = node_accuracy_error(output_unnormed["M"].to(self.device), target_unnormed["M"], accuracy_threshold=0.1*self.config['dataset']['pred']['target_std_M'])
                
                accuracy_loss_disp += accuracy_loss_disp_batch / num_disp
                accuracy_loss_N += accuracy_loss_N_batch / num_N
                accuracy_loss_M += accuracy_loss_M_batch / num_M
                
                relerror_loss_disp += relerror_loss_disp_batch / num_disp
                relerror_loss_N += relerror_loss_N_batch / num_N
                relerror_loss_M += relerror_loss_M_batch / num_M

                pbar.set_description(
                    f'Prediction - '
                    f'MAE Disp: {mae_loss_disp.item()/(batch_idx+1):.6f}, '
                    f'N: {mae_loss_N.item()/(batch_idx+1):.6f}, M: {mae_loss_M.item()/(batch_idx+1):.6f}, '
                    f'Acc Disp: {accuracy_loss_disp.item() / (batch_idx+1):.6f}, '
                    f'Acc N: {accuracy_loss_N.item() / (batch_idx+1):.6f}, '
                    f'Acc M: {accuracy_loss_M.item() / (batch_idx+1):.6f}, '
                    f'Rel Disp: {relerror_loss_disp.item() / (batch_idx+1):.6f}, '
                    f'Rel N: {relerror_loss_N.item() / (batch_idx+1):.6f}, '
                    f'Rel M: {relerror_loss_M.item() / (batch_idx+1):.6f}'
                )

            # Calculate average losses over the entire validation set - len(val_loader) is the number of batches
            total_mae_disp = mae_loss_disp.item() / len(pred_loader)
            total_mse_disp = mse_loss_disp.item() / len(pred_loader)

            total_mae_N = mae_loss_N.item() / len(pred_loader)
            total_mse_N = mse_loss_N.item() / len(pred_loader)

            total_mae_M = mae_loss_M.item() / len(pred_loader)
            total_mse_M = mse_loss_M.item() / len(pred_loader)

            total_accuracy_loss_disp = accuracy_loss_disp / len(pred_loader)
            total_accuracy_loss_N = accuracy_loss_N / len(pred_loader)
            total_accuracy_loss_M = accuracy_loss_M / len(pred_loader)

            total_relerror_loss_disp = relerror_loss_disp / len(pred_loader)
            total_relerror_loss_N = relerror_loss_N / len(pred_loader)
            total_relerror_loss_M = relerror_loss_M / len(pred_loader)

            if not self.debug:
                metrics = {
                    f"mae_disp": total_mae_disp,
                    f"mse_disp": total_mse_disp,
                    f"mae_N": total_mae_N,
                    f"mse_N": total_mse_N,
                    f"mae_M": total_mae_M,
                    f"mse_M": total_mse_M,
                    f"accuracy_loss_disp": total_accuracy_loss_disp,
                    f"accuracy_loss_N": total_accuracy_loss_N,
                    f"accuracy_loss_M": total_accuracy_loss_M,
                    f"relerror_loss_disp": total_relerror_loss_disp,
                    f"relerror_loss_N": total_relerror_loss_N,
                    f"relerror_loss_M": total_relerror_loss_M,
                    "pred/batch_run_time": current_run_time, 
                }
                if self.config['logger'] == 'comet':
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
            if self.config['logger'] == 'comet':
                self.writer.log_metrics(metrics)
        pbar.close()
        print("\nInvariance results:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

        return metrics