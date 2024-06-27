import os
import sys
import torch
import torch.nn as nn
from typing import Tuple
from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetFPModule

class PointNet2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[0, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 512, 1024], use_xyz=True
            )
        )

    def _break_up_pc(self, pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xyz = pc[..., :3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, data):
        x = data
        B, T, N, _ = x.shape
        x = x.reshape(-1, N, 3)
        xyz, features = self._break_up_pc(x)
        for module in self.SA_modules:
            xyz, features = module(xyz, features)     
        features = features.squeeze(-1).reshape(B, T, -1)
        return features

class PointNet2Regressor(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[feature_dim, 32, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=0.8,
                nsample=32,
                mlp=[512, 512, 512, 1024],
                use_xyz=True,
            )
        )
        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[256 + feature_dim, 256, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 128, 512, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 512, 512]))
        self.FP_modules.append(PointnetFPModule(mlp=[1024 + 512, 512, 512]))

    def _break_up_pc(self, pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xyz = pc[..., :3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, pointcloud):
        r"""
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        pointcloud = pointcloud.reshape(-1, *pointcloud.shape[-2:])
        xyz, features = self._break_up_pc(pointcloud)
        l_xyz, l_features = [xyz], [features]
        
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1],
                l_xyz[i], l_features[i - 1],
                l_features[i]
            )
        return l_features[0]

if __name__ == '__main__':
    encoder = PointNet2Encoder().cuda()
    input_pointclouds = torch.rand((1, 4, 256, 3)).cuda()
    output_pointclouds = encoder(input_pointclouds)
    print(output_pointclouds.shape)
    


