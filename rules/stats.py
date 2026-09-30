import torch
import torch.nn as nn

import numpy as np
from scipy.stats import skew, hmean


def compute_round_stats(grads, eps=1e-12):
    """
    grads shape:
        (n_clients, d)
    """

    n_clients = grads.shape[0]

    mean_vals = np.zeros(n_clients)
    std_vals = np.zeros(n_clients)
    skew_vals = np.zeros(n_clients)

    am_vals = np.zeros(n_clients)
    hm_vals = np.zeros(n_clients)
    ratio_vals = np.zeros(n_clients)

    l2_vals = np.zeros(n_clients)

    for i in range(n_clients):

        g = grads[i]

        g_abs = np.abs(g)

        mean_vals[i] = np.mean(g)
        std_vals[i] = np.std(g)
        skew_vals[i] = skew(g)

        am_vals[i] = np.mean(g_abs)
        hm_vals[i] = hmean(g_abs + eps)

        ratio_vals[i] = hm_vals[i] / (am_vals[i] + eps)

        l2_vals[i] = np.linalg.norm(g)

    # -------- Global Statistics --------

    all_grads = grads.reshape(-1)
    all_abs = np.abs(all_grads)

    global_am = np.mean(all_abs)
    global_hm = hmean(all_abs + eps)

    stats = {
        "mean": mean_vals,
        "std": std_vals,
        "skewness": skew_vals,
        "am": am_vals,
        "hm": hm_vals,
        "hm_am_ratio": ratio_vals,
        "l2_norm": l2_vals,

        "global_mean": np.mean(all_grads),
        "global_std": np.std(all_grads),
        "global_skewness": skew(all_grads),
        "global_am": global_am,
        "global_hm": global_hm,
        "global_hm_am_ratio":
            global_hm / (global_am + eps),
        "global_l2_norm":
            np.linalg.norm(all_grads)
    }

    return stats


def adaptor(input):

    """
    input:
        (1,d,n_clients)
    """

    x = input.squeeze(0)
    x = x.permute(1,0)

    # x -> (n_clients,d)

    grads_np = x.detach().cpu().numpy()

    round_stats = compute_round_stats(grads_np)

    # FedAvg

    out = torch.mean(x, dim=0, keepdim=True).T

    return out, round_stats


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

    def forward(self, input):

        out, round_stats = adaptor(input)

        return out, round_stats