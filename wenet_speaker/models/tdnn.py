#!/user/bin/env python3 

# Author: wsstriving@gmail.com (Shuai Wang)
"""TDNN model for x-vector learning"""
import torch
import torch.nn as nn
import torch.nn.functional as F 

from .pooling_layers import *

class TdnnLayer(nn.Module):
    def __init__(self, in_dim, out_dim, context_size, dilation=1, padding=0):
        """Define the TDNN layer, essentially 1-D convolution

        Args:
            in_dim (int): input dimension
            out_dim (int): output channels
            context_size (int): context size, essentially the filter size
            dilation (int, optional):  Defaults to 1.
            padding (int, optional):  Defaults to 0.
        """
        super(TdnnLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.context_size = context_size
        self.dilation = dilation
        self.padding = padding
        self.conv_1d = nn.Conv1d(self.in_dim, self.out_dim, self.context_size, 
                                dilation=self.dilation, padding=self.padding)
        
        # Set Affine=false to be compatible with the original kaldi version
        self.bn = nn.BatchNorm1d(out_dim, affine=False)

    def forward(self, x):
        out = self.conv_1d(x)
        out = F.relu(out)
        out = self.bn(out)
        return out


class Tdnn(nn.Module):
    def __init__(self, feat_dim=40, hid_dim=512, stats_dim=1500, n_stats=2, embed_dim=512):
        """
        Implementation of Kaldi style xvec, as described in
        X-VECTORS: ROBUST DNN EMBEDDINGS FOR SPEAKER RECOGNITION
        """
        super(Tdnn, self).__init__()
        self.feat_dim = feat_dim
        self.stats_dim = stats_dim
        self.n_stats = n_stats
        self.embed_dim = embed_dim
        
        self.frame_1 = TdnnLayer(feat_dim, hid_dim, context_size=5, dilation=1)
        self.frame_2 = TdnnLayer(hid_dim, hid_dim, context_size=3, dilation=2)
        self.frame_3 = TdnnLayer(hid_dim, hid_dim, context_size=3, dilation=3)
        self.frame_4 = TdnnLayer(hid_dim, hid_dim, context_size=1, dilation=1)
        self.frame_5 = TdnnLayer(hid_dim, stats_dim, context_size=1, dilation=1)
        self.pool = TSTP()
        self.seg_1 = nn.Linear(stats_dim * n_stats, embed_dim)
        self.seg_bn_1 = nn.BatchNorm1d(embed_dim, affine=False)
        self.seg_2 = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        out = self.frame_1(x)
        out = self.frame_2(out)
        out = self.frame_3(out)
        out = self.frame_4(out)
        out = self.frame_5(out)

        if isinstance(self.pool, SAP):
            stats, penalty = self.pool(out)
        else:
            stats = self.pool(out)

        embed_a = self.seg_1(stats)
        out = F.relu(embed_a)
        out = self.seg_bn_1(out)
        embed_b = self.seg_2(out)

        if isinstance(self.pool, SAP):
            return embed_a, embed_b, penalty
        else:
            return embed_a, embed_b


def test():
    net = Tdnn(40)
    y = net(torch.rand(10,40,200), torch.rand(10, 186))
    print(y[0].size(), y[2])
