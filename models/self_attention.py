# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import torch
from torch import nn
from math import sqrt

class SelfAttention(nn.Module):
    def __init__(self, dim_q = 1024, dim_k = 1024, dim_v =1024):
        super(SelfAttention, self).__init__()
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v
 
        self.linear_q = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_q, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k)
 
    def forward(self, x):
        batch, n, dim_q = x.shape
        assert dim_q == self.dim_q

        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact
        dist = torch.softmax(dist, dim=-1)
        att = torch.bmm(dist, v)
        
        return att

if __name__ == '__main__' :
    model = SelfAttention()
    fea = torch.rand((1, 3, 1024))
    
    x = model(fea)
    print(x.shape)