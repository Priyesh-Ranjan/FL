import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_similarity_matrix(grads, eps=1e-12):
    """
    Pairwise cosine similarity between client update vectors.

    Parameters
    ----------
    grads : torch.Tensor
        Shape [num_clients, vector_dimension].

    Returns
    -------
    torch.Tensor
        Shape [num_clients, num_clients].
    """
    normalized = F.normalize(
        grads,
        p=2,
        dim=1,
        eps=eps,
    )

    cs = normalized @ normalized.t()

    # Avoid tiny numerical values slightly outside [-1, 1].
    return torch.clamp(cs, -1.0, 1.0)


def fedavg_with_cosine(grads):
    """
    Compute cosine similarity for analysis, then perform vanilla FedAvg.

    Cosine similarity DOES NOT affect the aggregation weights.
    """
    n_clients = grads.shape[0]

    cs = cosine_similarity_matrix(grads)

    weights = torch.full(
        (n_clients,),
        1.0 / n_clients,
        dtype=grads.dtype,
        device=grads.device,
    )

    aggregated = torch.sum(
        grads * weights.view(-1, 1),
        dim=0,
    )

    return aggregated, weights, cs


def adaptor(input):
    """
    Input
    -----
    input : torch.Tensor
        Shape [1, vector_dimension, num_clients].

    Output
    ------
    out : torch.Tensor
        FedAvg update with shape [vector_dimension, 1].
    w : torch.Tensor
        Uniform FedAvg weights with shape [num_clients].
    cs : torch.Tensor
        Pairwise cosine similarity with shape
        [num_clients, num_clients].
    """
    x = input.squeeze(0)        # [d, N]
    client_updates = x.t()     # [N, d]

    out, w, cs = fedavg_with_cosine(client_updates)

    return out.unsqueeze(1), w, cs


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, input):
        return adaptor(input)
