import torch
import torch.nn as nn
import torch.nn.functional as F

def scatter(h, batch, dim=0, reduce="add"):
    """
    Aggregates node features based on batch assignments, supporting several reduction operations.
    
    Parameters:
        h (Tensor): The input node feature matrix (nodes by features).
        batch (LongTensor): A vector assigning each node to a batch. Nodes with the same batch
                            number will be aggregated together.
        dim (int, optional): The dimension over which to perform the scatter operation.
                             Typically, this is the node dimension.
        reduce (str, optional): The reduction operation to perform ("sum", "mean", "min", "max").
    
    Returns:
        Tensor: The aggregated node features.
    """
    unique_batches = torch.unique(batch)
    output_size = [h.size(i) if i != dim else len(unique_batches) for i in range(h.dim())]
    output = h.new_zeros(output_size)
    
    for b in unique_batches:
        mask = batch == b
        if reduce == "add":
            output[b] = h[mask].sum(dim=dim)
        elif reduce == "mean":
            output[b] = h[mask].mean(dim=dim)
        elif reduce == "min":
            output[b], _ = h[mask].min(dim=dim)
        elif reduce == "max":
            output[b], _ = h[mask].max(dim=dim)
        else:
            raise ValueError(f"Unsupported reduce operation: {reduce}")
    
    return output

def dropout_edge(edge_index, p=0.5, training=False):
    """Randomly remove edges from the edge list."""
    if not training:
        return edge_index
    num_edges = edge_index.size(1)
    mask = torch.rand(num_edges) > p
    edge_index = edge_index[:, mask]
    return edge_index

def radius_graph(x, r, batch=None, max_num_neighbors=None):
    """Constructs a graph based on vertex proximity."""
    distance_matrix = torch.cdist(x, x, p=2)
    adj_matrix = distance_matrix <= r

    if batch is not None:
        batch_x = batch.unsqueeze(0) == batch.unsqueeze(1)
        adj_matrix = adj_matrix * batch_x
    
    edge_index = adj_matrix.nonzero(as_tuple=False).t().contiguous()

    if max_num_neighbors is not None:
        raise NotImplementedError("max_num_neighbors is not implemented yet")
    
    return edge_index

def get_pbc_distances(x, edge_index, cell, cell_offsets, neighbors):
    """Compute the pairwise distances of a set of points, taking into account
    the periodic boundary conditions."""
    row, col = edge_index

    distance_vectors = x[row] - x[col]
    neighbors = neighbors.to(cell.device)
    cell = torch.repeat_interleave(cell, neighbors, dim=0)
    offsets = cell_offsets.float().view(-1, 1, 3).bmm(cell.float()).view(-1, 3)
    distance_vectors += offsets

    # compute distances
    distances = distance_vectors.norm(dim=-1)

    # redundancy: remove zero distances
    nonzero_idx = torch.arange(len(distances), device=distances.device)[distances != 0]
    edge_index = edge_index[:, nonzero_idx]
    distances = distances[nonzero_idx]

    out = {
        "edge_index": edge_index,
        "distances": distances,
    }

    out["distance_vec"] = distance_vectors[nonzero_idx]

    return out

class GraphNorm(nn.Module):
    """Graph normalization layer"""

    def __init__(self, in_channels):
        super(GraphNorm, self).__init__()
        self.in_channels = in_channels
        self.weight = nn.Parameter(torch.Tensor(in_channels))
        self.bias = nn.Parameter(torch.Tensor(in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = x - scatter(x, batch, dim=0, reduce="mean")
        var = scatter(x ** 2, batch, dim=0, reduce="mean")
        var = torch.sqrt(var + 1e-5)
        x = x / var
        x = x * self.weight + self.bias
        return x

    def __repr__(self):
        return "{}(in_channels={})".format(self.__class__.__name__, self.in_channels)

class GaussianSmearing(nn.Module):
    """Smears a distance distribution by a Gaussian function."""

    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

class MessagePassing(nn.Module):
    """Base class for message passing in GNNs."""
    def __init__(self):
        super(MessagePassing, self).__init__()

    def message(self, x_j, W): # Implemented in derived class
        raise NotImplementedError

    def propagate(self, edge_index, x, W):
        source, target = edge_index
        messages = self.message(x[target], W)

        aggr_messages = torch.zeros_like(x)
        aggr_messages.index_add_(0, target, messages)

        return aggr_messages