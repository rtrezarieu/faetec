from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from .gnn_utils import MessagePassing, GraphNorm, GaussianSmearing, dropout_edge, radius_graph, get_pbc_distances, scatter
from .modules.phys_embedding import PhysEmbedding
from src.force_decoder import ForceDecoder

def swish(x):
    return x * torch.sigmoid(x)

class FAENet(nn.Module):
    def __init__(self, cutoff=6.0, use_pbc=True, max_num_neighbors=40,
                 num_gaussians=1, num_filters=128, hidden_channels=128,   # num_gaussians = 50 if RBF active
                 tag_hidden_channels=32, pg_hidden_channels=32, 
                 phys_embeds=True, num_interactions=4,
                 output_disp=3, output_N=18, output_M=18,
                 act: str = "swish",
                 regress_forces: Optional[str] = None,
                 force_decoder_type: Optional[str] = "mlp", 
                 force_decoder_model_config: Optional[dict] = {"hidden_channels": 128}, **kwargs,    ### change 128 to 384?
                 ):
        super().__init__()

        self.cutoff = cutoff
        self.hidden_channels = hidden_channels
        self.max_num_neighbors = max_num_neighbors
        self.num_filters = num_filters
        self.num_gaussians = num_gaussians
        self.num_interactions = num_interactions
        self.pg_hidden_channels = pg_hidden_channels
        self.phys_embeds = phys_embeds
        self.tag_hidden_channels = tag_hidden_channels
        self.use_pbc = use_pbc

        self.dropout_edge = float(kwargs.get("dropout_edge") or 0)
        self.dropout_lin = float(kwargs.get("dropout_lin") or 0)

        self.regress_forces = regress_forces
        self.force_decoder_type = force_decoder_type
        self.force_decoder_model_config = force_decoder_model_config
        self.act = act
        self.output_disp = output_disp
        self.output_N = output_N
        self.output_M = output_M

        # Gaussian Basis
        # self.distance_expansion = GaussianSmearing(0.0, self.cutoff, self.num_gaussians)   # To reactivate if use of RBF: outputs of dimension self.num_gaussians 

        # Embedding block
        self.embed_block = EmbeddingBlock(
            self.num_gaussians,
            self.num_filters,
            self.hidden_channels,
            self.tag_hidden_channels,
            self.pg_hidden_channels,
            self.phys_embeds,
        )

        # Interaction block
        self.interaction_blocks = nn.ModuleList(
            [
                InteractionBlock(
                    self.hidden_channels,
                    self.num_filters,
                    0
                )
                for i in range(self.num_interactions)
            ]
        )

        # Output block
        # self.output_block = OutputBlock(self.hidden_channels, self.dropout_lin)

        output_types = ["disp", "N", "M"]
        self.output_blocks = {}
        for output_type in output_types:
            self.output_blocks[output_type] = (
                ForceDecoder(
                    self.force_decoder_type,
                    self.hidden_channels,  # 128 or 384 (config)
                    self.force_decoder_model_config,
                    self.act,
                    output_type,  # "disp", "N", "M"
                )
                if "direct" in self.regress_forces
                else None
            )

        # # Skip co // for energy
        # self.mlp_skip_co = nn.Linear((self.num_interactions + 1), 1)

        
    # conserver les modifs faites avec ajouts forces
    def forward(self, data):
        # z = data.atomic_numbers.long()
        pos = data.pos
        f = data.forces
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

        h, e = self.embed_block(f, f_norm, rel_pos, edge_length) ### au lieu de z, dat.tags edge_weight au lieu de edge_attr


        # Now change the predictions from energy only, to B, M, N // modify the output block
        for _, interaction in enumerate(self.interaction_blocks):
            # energy_skip_co.append(self.output_block(h, edge_index, edge_length, batch, data))
            h = interaction(h, edge_index, e)

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
    def __init__(self, num_gaussians, num_filters, hidden_channels,
                 tag_hidden_channels, pg_hidden_channels, phys_embeds,
                 ):
        super().__init__()
        # Physics 
        self.phys_emb = PhysEmbedding(props=phys_embeds, pg=True)
        phys_hidden_channels = self.phys_emb.n_properties

        # Period + group embeddings
        self.period_embedding = nn.Embedding(self.phys_emb.period_size, pg_hidden_channels)
        self.group_embedding = nn.Embedding(self.phys_emb.group_size, pg_hidden_channels)

        # Tag embedding - adsorbate, surface catalyst, or deeper catalyst
        self.tag_embedding = nn.Embedding(3, tag_hidden_channels)

        # Global embedding - only 85 elements are used in the dataset
        self.emb = nn.Embedding(
            85,
            hidden_channels
            - tag_hidden_channels
            - phys_hidden_channels
            - 2 * pg_hidden_channels,
        )

        # MLP
        self.lin = nn.Linear(hidden_channels, hidden_channels)

        # Edge embedding
        self.lin_e1 = nn.Linear(3, num_filters // 2)  # r_ij on the schema
        self.lin_e12 = nn.Linear(num_gaussians + 2, num_filters - (num_filters // 2))  # d_ij, +2 for Beam/Column One-Hot vectors

        self.lin_h1 = nn.Linear(3, hidden_channels // 2)  # r_ij on the schema
        self.lin_h12 = nn.Linear(num_gaussians, hidden_channels - (hidden_channels // 2)) # num_gaussians because the data went through RBF already

        self.emb.reset_parameters()
        self.tag_embedding.reset_parameters()
        self.period_embedding.reset_parameters()
        self.group_embedding.reset_parameters()
        nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin_e1.weight)
        self.lin_e1.bias.data.fill_(0)

    def forward(self, f, f_norm, rel_pos, edge_attr, tag=None): # z
        # Edge embedding
        rel_pos = self.lin_e1(rel_pos)  # r_ij
        edge_attr = self.lin_e12(edge_attr)  # d_ij
        e = torch.cat((rel_pos, edge_attr), dim=1)
        e = swish(e) 
        

        f = self.lin_h1(f) # f_i
        f_norm = self.lin_h12(f_norm)  # ||f_i||
        h = torch.cat((f, f_norm), dim=1)
        h = swish(h) 


        # Node embedding
        # Create atom embeddings based on its characteristic number
        # h = self.emb(z)

        # if self.phys_emb.device != h.device:
        #     self.phys_emb = self.phys_emb.to(h.device)

        # # Concat tag embedding
        # h_tag = self.tag_embedding(tag)
        # h = torch.cat((h, h_tag), dim=1)

        # # Concat physics embeddings
        # h_phys = self.phys_emb.properties[z]
        # h = torch.cat((h, h_phys), dim=1)

        # # Concat period & group embedding
        # h_period = self.period_embedding(self.phys_emb.period[z])
        # h_group = self.group_embedding(self.phys_emb.group[z])
        # h = torch.cat((h, h_period, h_group), dim=1)

        # MLP
        # h = swish(self.lin(h)) # We keep this MLP, for h. h stores the information while e filters? TO TEST

        return h, e

class InteractionBlock(MessagePassing):
    def __init__(self, hidden_channels, num_filters, dropout, ):
        super(InteractionBlock, self).__init__()
        self.hidden_channels = hidden_channels
        self.dropout = float(dropout)

        self.graph_norm = GraphNorm(hidden_channels)

        self.lin_geom = nn.Linear(num_filters + 2 * hidden_channels, hidden_channels)
        self.lin_h = nn.Linear(hidden_channels, hidden_channels)
        self.other_mlp = nn.Linear(hidden_channels, hidden_channels)

        nn.init.xavier_uniform_(self.lin_geom.weight)
        self.lin_geom.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.other_mlp.weight)
        self.other_mlp.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin_h.weight)
        self.lin_h.bias.data.fill_(0)

    def forward(self, h, edge_index, e):
        # Edge embedding
        if self.dropout > 0:
            h = F.dropout(h, p=self.dropout, training=self.training)
        e = torch.cat([e, h[edge_index[0]], h[edge_index[1]]], dim=1)
        e = swish(self.lin_geom(e)) # fij

        # Message passing
        h = self.propagate(edge_index, x=h, W=e) # somme des hj * fij
        h = swish(self.graph_norm(h)) # Pourquoi normalisation à cet endroit? 
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = swish(self.lin_h(h)) # 1ère couche du MLP
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = swish(self.other_mlp(h)) # 2nde couche du MLP

        return h

    def message(self, x_j, W):
        return x_j * W        # hj * fij produit terme à terme

# class OutputBlock(nn.Module):
#     def __init__(self, hidden_channels, dropout):
#         super().__init__()
#         self.dropout = float(dropout)
#         self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)  # MLP à 2 couches, la dimension intermédiaire = dimension d'entrée divisée par 2
#         self.lin2 = nn.Linear(hidden_channels // 2, 1)
#         self.w_lin = nn.Linear(hidden_channels, 1)

#     def forward(self, h, edge_index, edge_weight, batch, data=None):
#         alpha = self.w_lin(h)

#         # MLP
#         h = F.dropout(h, p=self.dropout, training=self.training)
#         h = self.lin1(h)
#         h = swish(h)
#         h = F.dropout(h, p=self.dropout, training=self.training)
#         h = self.lin2(h)
#         h = h * alpha

#         # Pooling
#         out = scatter(h, batch, dim=0, reduce="add")

#         return out
    


## ça ne convient pas pour les forces // pas les bonnes sorties
class OutputBlock(nn.Module):
    """Compute task-specific predictions from final atom representations."""

    def __init__(self, energy_head, hidden_channels, act, out_dim=1):    ### rajouter act = swish dans l'appel
        super().__init__()
        self.energy_head = energy_head
        self.act = act

        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = nn.Linear(hidden_channels // 2, out_dim)

        # if self.energy_head == "weighted-av-final-embeds":
        #     self.w_lin = nn.Linear(hidden_channels, 1)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        # if self.energy_head == "weighted-av-final-embeds":
        #     nn.init.xavier_uniform_(self.w_lin.weight)
        #     self.w_lin.bias.data.fill_(0)

    def forward(self, h, edge_index, edge_weight, batch, alpha):
        """Forward pass of the Output block.
        Called in FAENet to make prediction from final atom representations.

        Args:
            h (tensor): atom representations. (num_atoms, hidden_channels)
            edge_index (tensor): adjacency matrix. (2, num_edges)
            edge_weight (tensor): edge weights. (num_edges, )
            batch (tensor): batch indices. (num_atoms, )
            alpha (tensor): atom attention weights for late energy head. (num_atoms, )

        Returns:
            (tensor): graph-level representation (e.g. energy prediction)
        """
        # if self.energy_head == "weighted-av-final-embeds":
        #     alpha = self.w_lin(h)

        # MLP
        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        # if self.energy_head in {
        #     "weighted-av-initial-embeds",
        #     "weighted-av-final-embeds",
        # }:
        #     h = h * alpha

        # Global pooling
        # This is the sum_j operator on the pink graph
        out = scatter(h, batch, dim=0, reduce="add")   ### we need to sum up the predictions across the nodes when we want to predict the energy: which is one value per structure (and not per node)

        return out
