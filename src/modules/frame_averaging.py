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
            data.fa_pos, data.fa_f, data.fa_rot, data.lmbda_f = frame_averaging(
                data.pos, data.forces, self.fa_type, oc20=self.oc20
            )
            return data
        
def frame_averaging(pos, f, fa_method="stochastic", oc20=False):
    if oc20:
        used_pos = pos[:, :2]  ######## à changer pour translater les 3 corrdonées
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

    signs = list(product([1, -1], repeat=relative_pos.shape[1] + 1))
    basis_projections = [torch.tensor(x) for x in signs] # 16 combinations (or less for 2D)
    lmbda_f = torch.max(torch.norm(f, dim=-1, keepdim=True))
    fa_poss = []
    fa_cells = []
    fa_fs = []
    all_rots = []
    lmbda_fs = []

    
    for pm in basis_projections:  # pm for plus-minus # pm is one combination of the frame
        if oc20:
            new_eigenvec = pm[:2] * eigenvec  # Change the basis of the frame's element for 2D
            new_lmbda_f = lmbda_f * pm[2]  ###################################### à vérifier +++++
            fa_pos = relative_pos @ new_eigenvec  # Project the positions on the new basis
            # fa_f = f[:, :2] @ new_eigenvec / new_lmbda_f  # Adjust forces for 2D
            # fa_f = torch.cat((fa_f[:, :2], f[:, 2].unsqueeze(1)), dim=1)
            fa_f = f[:, :3:2] @ new_eigenvec / new_lmbda_f  # Adjust forces for 2D
            fa_f = torch.cat((fa_f[:, :1], f[:, 1].unsqueeze(1), fa_f[:, 1:]), dim=1)
            ########################## rebuild fa_f properly!!!!! with the last coordinates
        else:
            new_eigenvec = pm[:3] * eigenvec  # Change the basis of the frame's element for 3D
            new_lmbda_f = lmbda_f * pm[3]  # Change the sign of the force for 3D
            fa_pos = relative_pos @ new_eigenvec  # Project the positions on the new basis
            fa_f = f @ new_eigenvec / new_lmbda_f  # Adjust forces for 3D

    ###############################
    # for pm in basis_projections: # pm for plus-minus # pm is one combination of the frame
    #     new_eigenvec = pm[:3] * eigenvec # Change the basis of the frame's element
    #     new_lmbda_f = lmbda_f * pm[3] # Change the sign of the force
        # fa_pos = relative_pos @ new_eigenvec # Project the positions on the new basis
        # fa_f = f @ new_eigenvec / new_lmbda_f
    ###############################

        if new_pos is not None:
            full_eigenvec = torch.eye(3)
            fa_pos = torch.cat((fa_pos, new_pos.unsqueeze(1)), dim=1)
            full_eigenvec[:2, :2] = new_eigenvec
            new_eigenvec = full_eigenvec

        fa_poss.append(fa_pos)
        fa_fs.append(fa_f)
        all_rots.append(new_eigenvec.unsqueeze(0))
        lmbda_fs.append(new_lmbda_f)

        # # Handle rare case where no R is positive orthogonal
        # if all_fa_pos == []:
        #     all_fa_pos.append(fa_pos)
        #     all_cell.append(fa_cell)
        #     all_rots.append(new_eigenvec.unsqueeze(0))

    if fa_method == "full":
        return fa_poss, fa_fs, all_rots, lmbda_fs
    else: # stochastic
        # index = torch.randint(0, len(fa_poss) - 1, (1,)) # la borne supérieure est exclue
        index = torch.randint(0, len(fa_poss), (1,))
        return [fa_poss[index]], [fa_fs[index]], [all_rots[index]], [lmbda_fs[index]]


def data_augmentation(g, oc20=True):
    if not oc20:
        # Random rotation around all axes
        rotater = GraphRotate(-180, 180, [0, 1, 2])
    else:
        # In OC20, we only rotate around the z-axis because the main axis is fixed.
        rotater = GraphRotate(-180, 180, [2])

    graph_rotated, _, _ = rotater(g)

    return graph_rotated
