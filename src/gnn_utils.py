import torch
import torch.nn as nn
import torch.nn.functional as F

def dropout_edge(edge_index, p=0.5, training=False):
    """Randomly remove edges from the edge list."""
    if not training:
        return edge_index
    num_edges = edge_index.size(1)
    mask = torch.rand(num_edges) > p
    edge_index = edge_index[:, mask]
    return edge_index

def node_accuracy_error(output, target, accuracy_threshold, disp=True):
    """
    Calculate relative accuracy between output and target using a threshold.
    Returns the sum of the relative accuracies and the number of elements used.
    """
    device = output.device
    # condition = torch.any(torch.abs(target) > accuracy_threshold, dim=1)
    condition = torch.abs(target) > accuracy_threshold


    ones = torch.ones(target.shape).to(device)[condition]
    zeros = torch.zeros(target.shape).to(device)[condition]
    relative_accuracy = torch.clamp(ones - torch.div(torch.abs(target[condition] - output[condition]), torch.abs(target[condition])), min=0, max=1)
    relative_error = torch.div(torch.abs(target[condition] - output[condition]), torch.abs(target[condition]))

    return relative_accuracy.sum(), relative_error.sum(), condition.sum()

def swish(x):
    """Swish activation function"""
    return torch.nn.functional.silu(x)

def relu(x):
    """ReLU activation function"""
    return torch.nn.functional.relu(x)