import torch
from copy import deepcopy
from itertools import product
from src.datasets.data_utils import GraphRotate

class FrameAveraging:
    def __init__(self, equivariance="", fa_type="", oc20=True):
        self.fa_type = fa_type
        self.equivariance = equivariance
        self.oc20 = oc20

        assert self.equivariance in {
            "",
            "frame_averaging",
            "data_augmentation",
        }, f"Invalid equivariance: {self.equivariance}, must be one of ['', 'frame_averaging', 'data_augmentation']"

        assert self.fa_type in {
            "",
            "stochastic",
            "full",
        }, f"Invalid frame averaging type: {self.fa_type}, must be one of ['', 'stochastic', 'full']"

    def __call__(self, data):
        if self.equivariance == "":
            return data
        elif self.equivariance == "data_augmentation":
            return data_augmentation(data, self.oc20)
        else:
            data.fa_pos, data.fa_cell = frame_averaging(
                data.pos, data.cell if hasattr(data, "cell") else None, self.fa_type, oc20=self.oc20
            )
            return data

def frame_averaging(pos, cell=None, fa_method="stochastic", oc20=True):
    if oc20:
        used_pos = pos[:, :2]
    else:
        used_pos = pos

    relative_pos = used_pos - used_pos.mean(dim=0, keepdim=True)
    C = torch.matmul(relative_pos.t(), relative_pos) # Covariance matrix

    eigenval, eigenvec = torch.linalg.eigh(C)
    idx = eigenval.argsort(descending=True)
    eigenval = eigenval[idx]
    eigenvec = eigenvec[:, idx]

    if not oc20:
        new_pos = None # 3D case already has all 3 coordinates
        # In 2D, we need to get back the missing z-coordinate
    else:
        new_pos = pos[:, 2]

    signs = list(product([1, -1], repeat=relative_pos.shape[1]))
    basis_projections = [torch.tensor(x) for x in signs] # 8 or 4 possible combinations
    fa_poss = []
    fa_cells = []

    fa_cell = deepcopy(cell)

    for pm in basis_projections:
        new_eigenvec = pm * eigenvec # Change the basis of the frame's element
        fa_pos = relative_pos @ new_eigenvec # Project the positions on the new basis

        if new_pos is not None:
            full_eigenvec = torch.eye(3)
            fa_pos = torch.cat((fa_pos, new_pos.unsqueeze(1)), dim=1)
            full_eigenvec[:2, :2] = new_eigenvec
            new_eigenvec = full_eigenvec

        if cell is not None:
            fa_cell = cell @ new_eigenvec

        fa_poss.append(fa_pos)
        fa_cells.append(fa_cell)

    if fa_method == "full":
        return fa_poss, fa_cells
    else: # stochastic
        index = torch.randint(0, len(fa_poss) - 1, (1,))
        return [fa_poss[index]], [fa_cells[index]]

def data_augmentation(g, oc20=True):
    if not oc20:
        # Random rotation around all axes
        rotater = GraphRotate(-180, 180, [0, 1, 2])
    else:
        # In OC20, we only rotate around the z-axis because the main axis is fixed.
        rotater = GraphRotate(-180, 180, [2])

    graph_rotated, _, _ = rotater(g)

    return graph_rotated
