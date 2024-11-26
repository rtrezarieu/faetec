from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.norm import GraphNorm

from .gnn_utils import dropout_edge, swish, relu
from src.force_decoder import ForceDecoder

class FAEtec(nn.Module):
    #### add description

    def __init__(
        self, 
        cutoff: float = 6.0,
        act: str = "swish",
        hidden_channels: int = 128,   ######### à vérifier
        num_interactions: int = 6,   ######################################### IMPORTANT: adapter ce nombre de couches au dataset
        num_gaussians: int = 1, # num_gaussians = 50 if RBF active
        num_filters: int = 128,
        second_layer_MLP: bool = True, ## tester avec et sans

        mp_type: str = "updownscale_base",   ### message passing type, à comprendre et à implémenter
        graph_norm: bool = True,
        complex_mp: bool = False,

        output_disp: int = 3, 
        output_N: int = 18, 
        output_M: int = 18,
        regress_forces: Optional[str] = None,
        force_decoder_type: Optional[str] = "mlp", 
        force_decoder_model_config: Optional[dict] = {"hidden_channels": 128}, **kwargs,    ### change 128 to 384?
    ):
        super().__init__()

        self.cutoff = cutoff
        self.act = act
        self.hidden_channels = hidden_channels
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.num_filters = num_filters
        self.second_layer_MLP = second_layer_MLP
        self.mp_type = mp_type
        self.graph_norm = graph_norm
        self.complex_mp = complex_mp
        self.output_disp = output_disp
        self.output_N = output_N
        self.output_M = output_M
        self.regress_forces = regress_forces
        self.force_decoder_type = force_decoder_type
        self.force_decoder_model_config = force_decoder_model_config
        self.dropout_edge = float(kwargs.get("dropout_edge") or 0)

        if not isinstance(self.regress_forces, str):
            assert self.regress_forces is False or self.regress_forces is None, (
                "regress_forces must be a string "
                + "('', 'direct', 'direct_with_gradient_target') or False or None"
            )
            self.regress_forces = ""
            
        self.act = (
            (getattr(nn.functional, self.act) if self.act != "swish" else swish)
            if isinstance(self.act, str)
            else self.act
        )
        assert callable(self.act), (
            "act must be a callable function or a string "
            + "describing that function in torch.nn.functional"
        )

        # Embedding block
        self.embed_block = EmbeddingBlock(
            self.num_gaussians,
            self.num_filters,
            self.hidden_channels,
            self.act,
            self.second_layer_MLP,
        )

        # Interaction block
        self.interaction_blocks = nn.ModuleList(
            [
                InteractionBlock(
                    self.hidden_channels,
                    self.num_filters,
                    self.act,
                    self.mp_type,
                    self.complex_mp,
                    self.graph_norm,
                )
                for _ in range(self.num_interactions)
            ]
        )
        
        # Output block
        output_types = ["disp", "N", "M"]
        self.output_blocks = {}
        for output_type in output_types:
            self.output_blocks[output_type] = (
                ForceDecoder(
                    self.force_decoder_type,
                    self.hidden_channels,
                    self.force_decoder_model_config,
                    self.act,
                    output_type,  # "disp", "N", "M"
                )
                if "direct" in self.regress_forces
                else None
            )


    def forward(self, data):
        # z = data.atomic_numbers.long()
        pos = data.pos
        f = data.forces
        a = data.supports.float()
        a = a.unsqueeze(1)
        batch = data.batch  # the batch attribute should be created by the DataLoader
        energy_skip_co = []

        f_norm = torch.norm(f, dim=-1, keepdim=True) # Stocker cette matrice. Ce n'est pas lamda_f. 
        # lambda_f sera soit le maximum de ces nombres là, soit l'inverse de ce nombre.Il faudra faire ça dans la partie graph creation, avant. Au même endroit où U est calculé
        
        edge_index = data.edge_index
        rel_pos = pos[edge_index[0]] - pos[edge_index[1]] # (num_edges, num_dimensions)   ##### à calculer plutôt dans la génération du dataset
        edge_length = rel_pos.norm(dim=-1) # (num_edges,) = edges lengths
        edge_length = edge_length.unsqueeze(1) # (num_edges, 1)

        # edge_attr/edge_length = self.distance_expansion(edge_weight) # RBF (num_edges, num_gaussians), put num_gaussians=1, if RBF is not used
        edge_attr = torch.cat((data.beam_col, edge_length), dim=1) # Add E, I, A to it ######## créer edge_attr plutôt dans la base de données // à concaténer plus tard

        if self.dropout_edge > 0:
            edge_index, edge_mask = dropout_edge(     # edge_mask is a boolean tensor of shape (num_edges,)
                edge_index,
                p=self.dropout_edge,
                training=self.training
            )
            edge_length = edge_length[edge_mask]
            edge_attr = edge_attr[edge_mask]
            rel_pos = rel_pos[edge_mask]

        h, e = self.embed_block(f, f_norm, a, rel_pos, edge_length) ### au lieu de z, dat.tags edge_weight au lieu de edge_attr


        # Now change the predictions from energy only, to B, M, N // modify the output block
        for _, interaction in enumerate(self.interaction_blocks):
            # energy_skip_co.append(self.output_block(h, edge_index, edge_length, batch, data))
            h = h + interaction(h, edge_index, e)

        # energy = self.output_block(h, edge_index, edge_length, batch, data=data)
        disp = self.output_blocks["disp"](h)
        forces = self.output_blocks["N"](h)
        moments = self.output_blocks["M"](h)


        # energy_skip_co.append(energy)
        # energy = self.mlp_skip_co(torch.cat(energy_skip_co, dim=1))

        preds = {
            # "energy": energy,    ######################### à modifier en suivant la version originale de FAENet
            "disp": disp,
            "N": forces,
            "M": moments
            # "hidden_state": h,
        }
        return preds


class EmbeddingBlock(nn.Module):
    """Initialise atom and edge representations."""

    def __init__(
        self, 
        num_gaussians, 
        num_filters, 
        hidden_channels,
        act,
        second_layer_MLP,
    ):
        super().__init__()
        self.act = act
        self.second_layer_MLP = second_layer_MLP

        # MLP
        # self.lin = nn.Linear(hidden_channels, hidden_channels)  # For edge embeddings
        # if self.second_layer_MLP:
        #     self.lin_2 = nn.Linear(hidden_channels, hidden_channels)

        # Edge embedding
        self.lin_e1 = nn.Linear(3, num_filters // 2)  # r_ij on the schema
        self.lin_e12 = nn.Linear(num_gaussians, num_filters - (num_filters // 2))  # d_ij, +2 for Beam/Column One-Hot vectors

        self.lin_h1 = nn.Linear(3, hidden_channels // 2)  # r_ij on the schema
        self.lin_h12 = nn.Linear(num_gaussians, hidden_channels - (hidden_channels // 2) - 4) # num_gaussians because the data went through RBF already

        self.lin_A = nn.Linear(1, 4)  # A_i   ######## définir 4 comme une dimension d'embeddigns des supports

        if self.second_layer_MLP:
            self.lin_e2 = nn.Linear(num_filters, num_filters)
            self.lin_h2 = nn.Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.lin.weight)
        # self.lin.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin_e1.weight)
        self.lin_e1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin_e12.weight)
        self.lin_e12.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin_h1.weight)
        self.lin_h1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin_h12.weight)
        self.lin_h12.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin_A.weight)
        self.lin_A.bias.data.fill_(0)
        if self.second_layer_MLP:
            # nn.init.xavier_uniform_(self.lin_2.weight)
            # self.lin_2.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.lin_e2.weight)
            self.lin_e2.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.lin_h2.weight)
            self.lin_h2.bias.data.fill_(0)

    def forward(self, f, f_norm, a, rel_pos, edge_attr, tag=None):
        # Edge embedding
        rel_pos = self.lin_e1(rel_pos)  # r_ij
        edge_attr = self.lin_e12(edge_attr)  # d_ij    
        e = torch.cat((rel_pos, edge_attr), dim=1)
        e = self.act(e)

        f = self.lin_h1(f) # f_i
        f_norm = self.lin_h12(f_norm)  # ||f_i||
        A = self.lin_A(a)
        h = torch.cat((f, f_norm, A), dim=1)    ################
        h = self.act(h)
        
        if self.second_layer_MLP:
            # e = self.lin_e2(e)
            e = self.act(self.lin_e2(e))
            # h = self.lin_h2(h)
            h = self.act(self.lin_h2(h))

        """
        Placeholder for edge embeddings
        """

        return h, e

class InteractionBlock(MessagePassing):
    """Updates atom representations through custom message passing."""

    def __init__(
        self,
        hidden_channels,
        num_filters,
        act,
        mp_type,
        complex_mp,
        graph_norm,
    ):
        super(InteractionBlock, self).__init__()
        self.act = act
        self.mp_type = mp_type
        self.hidden_channels = hidden_channels
        self.complex_mp = complex_mp
        self.graph_norm = graph_norm
        if graph_norm:
            self.graph_norm = GraphNorm(
                hidden_channels if "updown" not in self.mp_type else num_filters
            )
            # self.graph_norm = GraphNorm(hidden_channels)

        if self.mp_type == "simple":
            self.lin_h = nn.Linear(hidden_channels, hidden_channels)

        elif self.mp_type == "updownscale":
            self.lin_geom = nn.Linear(num_filters, num_filters)
            self.lin_down = nn.Linear(hidden_channels, num_filters)
            self.lin_up = nn.Linear(num_filters, hidden_channels)

        elif self.mp_type == "updownscale_base":
            self.lin_geom = nn.Linear(num_filters + 2 * hidden_channels, num_filters)
            self.lin_down = nn.Linear(hidden_channels, num_filters)
            self.lin_up = nn.Linear(num_filters, hidden_channels)

        elif self.mp_type == "updown_local_env":
            self.lin_down = nn.Linear(hidden_channels, num_filters)
            self.lin_geom = nn.Linear(num_filters, num_filters)
            self.lin_up = nn.Linear(2 * num_filters, hidden_channels)

        else:  # base
            self.lin_geom = nn.Linear(
                num_filters + 2 * hidden_channels, hidden_channels
            )
            self.lin_h = nn.Linear(hidden_channels, hidden_channels)

        if self.complex_mp:
            self.other_mlp = nn.Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        if self.mp_type != "simple":
            nn.init.xavier_uniform_(self.lin_geom.weight)
            self.lin_geom.bias.data.fill_(0)
        if self.complex_mp:
            nn.init.xavier_uniform_(self.other_mlp.weight)
            self.other_mlp.bias.data.fill_(0)
        if self.mp_type in {"updownscale", "updownscale_base", "updown_local_env"}:
            nn.init.xavier_uniform_(self.lin_up.weight)
            self.lin_up.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.lin_down.weight)
            self.lin_down.bias.data.fill_(0)
        else:
            nn.init.xavier_uniform_(self.lin_h.weight)
            self.lin_h.bias.data.fill_(0)

    def forward(self, h, edge_index, e):
        # Edge embedding
        if self.mp_type in {"base", "updownscale_base"}:
            e = torch.cat([e, h[edge_index[0]], h[edge_index[1]]], dim=1)

        if self.mp_type in {
            "updownscale",
            "base",
            "updownscale_base",
        }:
            e = self.act(self.lin_geom(e)) # fij graph convolution filter

        # --- Message Passing block --

        if self.mp_type == "updownscale" or self.mp_type == "updownscale_base":
            h = self.act(self.lin_down(h))  # downscale node rep.
            h = self.propagate(edge_index, x=h, W=e)  # propagate
            if self.graph_norm:
                h = self.act(self.graph_norm(h))
            h = self.act(self.lin_up(h))  # upscale node rep.

        elif self.mp_type == "updown_local_env":
            h = self.act(self.lin_down(h))
            chi = self.propagate(edge_index, x=h, W=e, local_env=True)
            e = self.lin_geom(e)
            h = self.propagate(edge_index, x=h, W=e)  # propagate
            if self.graph_norm:
                h = self.act(self.graph_norm(h))
            h = torch.cat((h, chi), dim=1)
            h = self.lin_up(h)

        elif self.mp_type in {"base", "simple"}:
            h = self.propagate(edge_index, x=h, W=e)  # propagate: sum of hj * fij
            if self.graph_norm:
                h = self.act(self.graph_norm(h))
            h = self.act(self.lin_h(h))

        else:
            raise ValueError("mp_type provided does not exist")
        
        # Message passing
        if self.complex_mp:
            h = self.act(self.other_mlp(h))

        return h

    def message(self, x_j, W, local_env=None):
        if local_env is not None:
            return W
        else:
            return x_j * W # hj * fij produit terme à terme     
