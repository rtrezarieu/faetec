import torch
from copy import deepcopy

def get_rotation_matrix(degrees, axis=0):
    rad_angle = torch.deg2rad(degrees)
    if axis == 0:
        rot_matrix = torch.tensor([[1, 0, 0],
                                    [0, torch.cos(rad_angle), -torch.sin(rad_angle)],
                                    [0, torch.sin(rad_angle), torch.cos(rad_angle)]])
    elif axis == 1:
        rot_matrix = torch.tensor([[torch.cos(rad_angle), 0, torch.sin(rad_angle)],
                                    [0, 1, 0],
                                    [-torch.sin(rad_angle), 0, torch.cos(rad_angle)]])
    elif axis == 2:
        rot_matrix = torch.tensor([[torch.cos(rad_angle), -torch.sin(rad_angle), 0],
                                    [torch.sin(rad_angle), torch.cos(rad_angle), 0],
                                    [0, 0, 1]])
    return rot_matrix

class GraphRotate():
    def __init__(self, min_degrees, max_degrees, axis=[0, 1, 2]):
        self.min_degrees = min_degrees
        self.max_degrees = max_degrees
        self.axis = axis

    def __call__(self, data):
        data = deepcopy(data)
        rotate_cell = hasattr(data, 'cell')

        rotation_matrix = torch.eye(3, device=data.pos.device, dtype=data.pos.dtype)
        for ax in self.axis:
            degrees = torch.randint(self.min_degrees, self.max_degrees, (1,))
            rotation_matrix = get_rotation_matrix(degrees, ax).to(data.pos.device, data.pos.dtype) @ rotation_matrix

        data.pos = data.pos @ rotation_matrix.to(data.pos.device, data.pos.dtype)
        if rotate_cell:
            data.cell = data.cell @ rotation_matrix.to(data.cell.device, data.cell.dtype)

        return data, rotation_matrix, torch.inverse(rotation_matrix)

class GraphReflect():
    # 2D reflections
    def __init__(self):
        pass

    def __call__(self, data):
        data = deepcopy(data)
        reflect_cell = hasattr(data, 'cell')

        reflect_matrix = torch.eye(3, device=data.pos.device, dtype=data.pos.dtype)
        reflection_type = torch.randint(0, 4, (1,))
        for ax in [0, 1]:
            if (reflection_type in [0, 1] and ax == 0) or (reflection_type in [0, 2] and ax == 1):
                reflect_matrix[ax, ax] = -1
            elif reflection_type == 3 and ax == 0:
                buffer = deepcopy(reflect_matrix[:, 0])
                reflect_matrix[:, 0], reflect_matrix[:, 1] = reflect_matrix[:, 1], buffer

        data.pos = data.pos @ reflect_matrix.to(data.pos.device, data.pos.dtype)
        if reflect_cell:
            data.cell = data.cell @ reflect_matrix.to(data.cell.device, data.cell.dtype)

        return data, reflect_matrix, torch.inverse(reflect_matrix)

class Normalizer(object):
    """Normalize a Tensor and restore it later."""

    def __init__(self, tensor=None, mean=None, std=None, device=None):
        """tensor is taken as a sample to calculate the mean and std"""
        if tensor is None and mean is None:
            return

        if device is None:
            device = "cpu"

        self.device = device

        if tensor is not None:
            self.mean = torch.mean(tensor, dim=0).to(device)
            self.std = torch.std(tensor, dim=0).to(device)
            return

        if mean is not None and std is not None:
            self.mean = torch.tensor(mean).to(device)
            self.std = torch.tensor(std).to(device)

        self.hof_mean = None
        self.hof_std = None
        self.rescale_with_hof = False

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        if self.hof_mean:
            self.hof_mean = self.hof_mean.to(device)
        if self.hof_std:
            self.hof_std = self.hof_std.to(device)
        self.device = device

    def norm(self, tensor, hofs=None):
        if hofs is not None and self.rescale_with_hof:
            return tensor / hofs - self.hof_mean
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor, hofs=None):
        if hofs is not None and self.rescale_with_hof:
            return (normed_tensor + self.hof_mean) * hofs
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        sd = {"mean": self.mean, "std": self.std}
        if self.rescale_with_hof:
            sd["hof_stats"] = {
                "mean": self.hof_mean,
                "std": self.hof_std,
            }
        return sd

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"].to(self.mean.device)
        self.std = state_dict["std"].to(self.mean.device)
        if "hof_stats" in state_dict:
            self.set_hof_rescales(state_dict["hof_stats"])

    def set_hof_rescales(self, hof_stats):
        self.hof_mean = torch.tensor(hof_stats["mean"], device=self.device)
        self.hof_std = torch.tensor(hof_stats["std"], device=self.device)
        self.rescale_with_hof = True
