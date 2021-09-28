#!/user/bin/env python3 

# Author: wsstriving@gmail.com (Shuai Wang)
"""Pooling functions to aggregate frame-level deep features
into segment-level speaker embeddings"""
import torch
import torch.nn as nn
import torch.nn.functional as F 


class TAP(nn.Module):
    """
    Temporal average pooling, only first-order mean is considered
    """    
    def __init__(self):
        super(TAP, self).__init__()
    
    def forward(self, x):
        pooling_mean = x.mean(dim=-1)
        # To be compatable with 2D input
        pooling_mean = pooling_mean.flatten(start_dim=1)
        return pooling_mean


class TSDP(nn.Module):
    """
    Temporal standard deviation pooling, only second-order std is considered
    """
    def __init__(self):
        super(TSDP, self).__init__()
    
    def forward(self, x):
        # The last dimension is the temporal axis
        pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-8)
        pooling_std = pooling_std.flatten(start_dim=1)
        return pooling_std


class TSTP(nn.Module):
    """
    Temporal statstics pooling, concatenate mean and std, which is used in x-vector
    Comment: simple concatenation can not make full use of both statistics
    """
    def __init__(self):
        super(TSTP, self).__init__()
    
    def forward(self, x):
        # The last dimension is the temporal axis
        pooling_mean = x.mean(dim=-1)
        pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-8)
        pooling_mean = pooling_mean.flatten(start_dim=1)
        pooling_std = pooling_std.flatten(start_dim=1)

        stats = torch.cat((pooling_mean, pooling_std), 1)
        return stats
