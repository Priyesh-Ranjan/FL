import torch
import torch.nn as nn

'''
FoolsGold

retrieved from https://github.com/DistributedML/FoolsGold/blob/master/deep-fg/fg/foolsgold.py

Reference:
Fung, Clement, Chris JM Yoon, and Ivan Beschastnikh. "The Limitations of Federated Learning in Sybil Settings." 23rd International Symposium on Research in Attacks, Intrusions and Defenses ({RAID} 2020). 2020.

'''

import numpy as np
from sklearn.metrics import mutual_info_score

# Takes in grad
# Compute similarity
# Get weightings

def foolsgold(grads):
    n_clients = grads.shape[0]
    cs = np.zeros((n_clients, n_clients))

    grads_disc = []
    n_bins=20
    for g in grads:
        hist, bin_edges = np.histogram(g, bins=n_bins)
        g_disc = np.digitize(g, bin_edges[:-1])
        grads_disc.append(g_disc)

    for i in range(n_clients):
        for j in range(n_clients):
            cs[i, j] = mutual_info_score(grads_disc[i], grads_disc[j])
    #cs = mutual_info_score(grads)#- np.eye(n_clients)
    cs_val = cs
    maxcs = np.max(cs, axis=1)
    # pardoning
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99

    # Logit function
    wv = (np.log(wv / (1 - wv)) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0
    wv = torch.ones(n_clients) / n_clients

    return wv, cs_val


def adaptor(input):
    '''
    compute foolsgold 
    
    input : 1* vector dimension * n
    
    return 
        foolsGold :  vector dimension 
    '''
    x = input.squeeze(0)
    x = x.permute(1, 0)
    w, cs = foolsgold(x)
    print(w)
    print(cs)
    w = w / w.sum()
    out = torch.sum(x.permute(1, 0) * w, dim=1, keepdim=True)
    return out, w, cs


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, input):
        #         print(input.shape)
        '''
        input: batchsize* vector dimension * n 
        (1 by d by n)
        
        return 
            out : size =vector dimension, will be flattened afterwards
        '''
        out, w, cs = adaptor(input)
        print(type(out))

        return out, w, cs
