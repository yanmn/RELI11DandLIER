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
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import torch
import os.path as osp
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from smpl import SMPL
from conf.model_file import SMPL_MEAN_PARAMS
from models.cross_attention import PositionwiseFeedForward
from models.self_attention import SelfAttention
from utils.geometry import rot6d_to_rotmat, rotation_matrix_to_angle_axis

class TemporalEncoder(nn.Module):
    def __init__(
            self,
            n_layers=2,
            hidden_size=1024,
            add_linear=True,
            bidirectional=False,
            use_residual=True
    ):
        super(TemporalEncoder, self).__init__()
        self.gru = nn.GRU(
            input_size=2048,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=n_layers
        )
        self.linear = None
        if bidirectional:
            self.linear = nn.Linear(hidden_size*2, 2048)
        elif add_linear:
            self.linear = nn.Linear(hidden_size, 2048)
        self.use_residual = use_residual

    def forward(self, x):
        n,t,f = x.shape
        x = x.permute(1,0,2)
        y, _ = self.gru(x)
        if self.linear:
            y = F.relu(y)
            y = self.linear(y.view(-1, y.size(-1)))
            y = y.view(t,n,f)
        if self.use_residual and y.shape[-1] == 2048:
            y = y + x
        y = y.permute(1,0,2)
        
        return y

def projection(pred_joints, pred_camera):
    pred_cam_t = torch.stack([pred_camera[:, 1],
                              pred_camera[:, 2],
                              2 * 5000. / (224. * pred_camera[:, 0] + 1e-9)], dim=-1)
    batch_size = pred_joints.shape[0]
    camera_center = torch.zeros(batch_size, 2)
    pred_keypoints_2d = perspective_projection(pred_joints,
                                               rotation=torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).to(pred_joints.device),
                                               translation=pred_cam_t,
                                               focal_length=5000.,
                                               camera_center=camera_center)
    
    pred_keypoints_2d = pred_keypoints_2d / (224. / 2.)
    return pred_keypoints_2d


def perspective_projection(points, rotation, translation, focal_length, camera_center):
    
    # This function computes the perspective projection of a set of points.
    # Input:
    #     points (bs, N, 3): 3D points
    #     rotation (bs, 3, 3): Camera rotation
    #     translation (bs, 3): Camera translation
    #     focal_length (bs,) or scalar: Focal length
    #     camera_center (bs, 2): Camera center
    
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length
    K[:,1,1] = focal_length
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center
    
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    projected_points = points / points[:,:,-1].unsqueeze(-1)
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]


class Regressor(nn.Module):
    def __init__(self, smpl_mean_params=SMPL_MEAN_PARAMS):
        super(Regressor, self).__init__()

        npose = 24 * 6
        self.smpl = SMPL().cuda()
        self.fc1 = nn.Linear(512 * 2 + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        batch_size = x.shape[0]
        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam

        for i in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
        
        pred_human_vertices = self.smpl(
            pose = pred_rotmat,
            beta = torch.zeros((batch_size, 10)).cuda()
        )
        
        pred_joints = self.smpl.get_full_joints(
            vertices = pred_human_vertices
        ).reshape(batch_size, 24, 3)

        pred_keypoints_2d = projection(pred_joints, pred_cam)

        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

        output = {'theta'  : torch.cat([pred_cam, pose, pred_shape], dim=1), # (B*T, 85) 85 = 3 + 72 + 10
                  'kp_2d'  : pred_keypoints_2d, # (B*T, 24, 2)
                  'kp_3d'  : pred_joints, # (B*T, 24, 3)
                  'rotmat' : pred_rotmat } # (B*T, 24, 3, 3)
        return output

class VIBE(nn.Module):
    def __init__(self,
                 n_layers=2,
                 hidden_size=1024,
                 add_linear=True,
                 bidirectional=False,
                 use_residual=True):
        super(VIBE, self).__init__()
        self.encoder = TemporalEncoder(
            n_layers=n_layers,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            add_linear=add_linear,
            use_residual=use_residual,
        )
        self.regressor = Regressor()
        self.cross_attention = nn.MultiheadAttention(embed_dim = 1024, num_heads = 4, dropout = 0.1)
        self.ffn = PositionwiseFeedForward()
        self.self_attention = nn.MultiheadAttention(embed_dim = 1024, num_heads = 4, dropout = 0.1)
        self.fc = nn.Linear(2048, 1024)
        self.norm = nn.LayerNorm(normalized_shape=1024)
        self.layer_norm = nn.LayerNorm(1024, eps=1e-6)
        self.trans_encoder1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model = 1024, nhead = 4),
            num_layers = 6)
        self.trans_encoder2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model = 1024, nhead = 4),
            num_layers = 12)

    def forward(self, input, pc_feature = None):
        batch_size, seqlen = input.shape[:2]
        feature = self.encoder(input)
        
        feature = feature.reshape(-1, feature.size(-1))
        img_feature = self.fc(feature)
        
        img_feature = img_feature.reshape(batch_size, seqlen, 1024)
        pc_res = pc_feature
        img_res = img_feature
            
        pc_feature = self.trans_encoder2(pc_feature)
        pc_feature,_ = self.self_attention(pc_feature, pc_feature, pc_feature)
        
        img_feature = self.trans_encoder1(img_feature)
        img_feature,_ = self.self_attention(img_feature, img_feature, img_feature)
        
        fusion_feature,_ = self.cross_attention(pc_feature, img_feature, img_feature)
        fusion_feature = self.ffn(fusion_feature)
        
        for _ in range(5):
            pc_feature,_ = self.self_attention(pc_feature, pc_feature, pc_feature)
            img_feature,_ = self.self_attention(img_feature, img_feature, img_feature)
            fusion_feature,_ = self.cross_attention(pc_feature, fusion_feature, img_feature)
            fusion_feature = self.ffn(fusion_feature)
        
        fusion_feature = fusion_feature + pc_res
        fusion_feature = self.layer_norm(fusion_feature)
        
        res_fusion = fusion_feature
        fusion_feature = self.ffn(fusion_feature)
        
        fusion_feature = fusion_feature + res_fusion
        fusion_feature = self.layer_norm(fusion_feature)
        
        fusion_feature = fusion_feature + pc_res + img_res
        
        fusion_feature = fusion_feature.reshape(-1, fusion_feature.size(-1)) # B*T, 1024
        # ------------------------------------------------
        smpl_output = self.regressor(fusion_feature)
        
        return feature, smpl_output
    
class VIBE2(nn.Module):
    def __init__(self,
                 n_layers=2,
                 hidden_size=1024,
                 add_linear=True,
                 bidirectional=False,
                 use_residual=True):
        super(VIBE2, self).__init__()
        self.encoder = TemporalEncoder(
            n_layers=n_layers,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            add_linear=add_linear,
            use_residual=use_residual,
        )
        self.regressor = Regressor()
        self.cross_attention = nn.MultiheadAttention(embed_dim = 1024, num_heads = 4, dropout = 0.1)
        self.ffn = PositionwiseFeedForward()
        self.self_attention = nn.MultiheadAttention(embed_dim = 1024, num_heads = 4, dropout = 0.1)
        self.fc = nn.Linear(2048, 1024)
        self.norm = nn.LayerNorm(normalized_shape=1024)
        self.layer_norm = nn.LayerNorm(1024, eps=1e-6)
        self.trans_encoder1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model = 1024, nhead = 4),
            num_layers = 6)
        self.trans_encoder2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model = 1024, nhead = 4),
            num_layers = 12)

    def forward(self, input, pc_feature = None):
        batch_size, seqlen = input.shape[:2]
        img_feature = input
        
        pc_res = pc_feature
        img_res = img_feature
            
        pc_feature = self.trans_encoder2(pc_feature)
        pc_feature,_ = self.self_attention(pc_feature, pc_feature, pc_feature)
        
        img_feature = self.trans_encoder1(img_feature)
        img_feature,_ = self.self_attention(img_feature, img_feature, img_feature)
        
        fusion_feature,_ = self.cross_attention(pc_feature, img_feature, img_feature)
        fusion_feature = self.ffn(fusion_feature)
        
        for _ in range(5):
            pc_feature,_ = self.self_attention(pc_feature, pc_feature, pc_feature)
            img_feature,_ = self.self_attention(img_feature, img_feature, img_feature)
            fusion_feature,_ = self.cross_attention(pc_feature, fusion_feature, img_feature)
            fusion_feature = self.ffn(fusion_feature)
        
        fusion_feature = fusion_feature + pc_res
        fusion_feature = self.layer_norm(fusion_feature)
        
        res_fusion = fusion_feature
        
        fusion_feature = self.ffn(fusion_feature)
        
        fusion_feature = fusion_feature + res_fusion
        fusion_feature = self.layer_norm(fusion_feature)
        
        fusion_feature = fusion_feature + pc_res + img_res
        
        fusion_feature = fusion_feature.reshape(-1, fusion_feature.size(-1)) # B*T, 1024
        # -------------------------------------------------
        smpl_output = self.regressor(fusion_feature)
        
        return img_feature, smpl_output

class VIBE_rgb(nn.Module):
    def __init__(self,
                 n_layers=2,
                 hidden_size=1024,
                 add_linear=True,
                 bidirectional=False,
                 use_residual=True):
        super(VIBE_rgb, self).__init__()
        self.encoder = TemporalEncoder(
            n_layers=n_layers,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            add_linear=add_linear,
            use_residual=use_residual,
        )
        self.regressor = Regressor()
        self.ffn = PositionwiseFeedForward()
        self.self_attention = SelfAttention()
        self.fc = nn.Linear(2048, 1024)
        self.norm = nn.LayerNorm(normalized_shape=1024)
        self.layer_norm = nn.LayerNorm(1024, eps=1e-6)

    def forward(self, input):
        batch_size, seqlen = input.shape[:2]
        feature = self.encoder(input)
        feature = feature.reshape(-1, feature.size(-1))

        img_feature = self.fc(feature)

        img_feature = img_feature.reshape(batch_size, seqlen, 1024)
        img_res = img_feature
        
        for _ in range(6):
            img_feature_self = self.self_attention(img_feature)
            img_feature = img_feature + img_feature_self
            img_feature = self.norm(img_feature)
            img_feature = self.ffn(img_feature)
        
        img_feature = self.self_attention(img_feature)
        fusion_feature = self.ffn(img_feature)
        
        for _ in range(5):
            img_feature = self.self_attention(img_feature)
            fusion_feature = self.ffn(img_feature)
        
        fusion_feature = fusion_feature + img_res
        fusion_feature = self.layer_norm(fusion_feature)
        
        res_fusion = fusion_feature
        
        fusion_feature = self.ffn(fusion_feature)
        
        fusion_feature = fusion_feature + res_fusion
        fusion_feature = self.layer_norm(fusion_feature)
        
        fusion_feature = fusion_feature + img_res
        fusion_feature = fusion_feature.reshape(-1, fusion_feature.size(-1)) # B*T, 1024
        
        smpl_output = self.regressor(fusion_feature)

        return feature, smpl_output

class VIBE_pc(nn.Module):
    def __init__(self,
                 n_layers=2,
                 hidden_size=1024,
                 add_linear=True,
                 bidirectional=False,
                 use_residual=True):
        super(VIBE_pc, self).__init__()
        self.regressor = Regressor()
        self.ffn = PositionwiseFeedForward()
        self.self_attention = SelfAttention()
        self.fc = nn.Linear(2048, 1024)
        self.norm = nn.LayerNorm(normalized_shape=1024)
        self.layer_norm = nn.LayerNorm(1024, eps=1e-6)

    def forward(self, pc_feature = None):
        pc_res = pc_feature
        
        for _ in range(6):
            pc_feature_self = self.self_attention(pc_feature)
            pc_feature = pc_feature + pc_feature_self
            pc_feature = self.norm(pc_feature)
            pc_feature = self.ffn(pc_feature)
            
        pc_feature = self.self_attention(pc_feature)
        fusion_feature = self.ffn(pc_feature)
        
        for _ in range(5):
            pc_feature = self.self_attention(pc_feature)
            fusion_feature = self.ffn(pc_feature)
        
        fusion_feature = fusion_feature + pc_res
        fusion_feature = self.layer_norm(fusion_feature)
        
        res_fusion = fusion_feature
        
        fusion_feature = self.ffn(fusion_feature)
        
        fusion_feature = fusion_feature + res_fusion
        fusion_feature = self.layer_norm(fusion_feature)
        
        fusion_feature = fusion_feature + pc_res
        
        fusion_feature = fusion_feature.reshape(-1, fusion_feature.size(-1))
        # --------------------------------------------
        smpl_output = self.regressor(fusion_feature)

        return smpl_output

class VIBE_event(nn.Module):
    def __init__(self,
                 n_layers=2,
                 hidden_size=1024,
                 add_linear=True,
                 bidirectional=False,
                 use_residual=True):
        super(VIBE_event, self).__init__()
        self.encoder = TemporalEncoder(
            n_layers=n_layers,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            add_linear=add_linear,
            use_residual=use_residual,
        )
        self.regressor = Regressor()
        self.ffn = PositionwiseFeedForward()
        self.self_attention = SelfAttention()
        self.fc = nn.Linear(2048, 1024)
        self.norm = nn.LayerNorm(normalized_shape=1024)
        self.layer_norm = nn.LayerNorm(1024, eps=1e-6)

    def forward(self, input):
        batch_size, seqlen = input.shape[:2]
        feature = self.encoder(input)
        
        feature = feature.reshape(-1, feature.size(-1))
        img_feature = self.fc(feature)
        
        img_feature = img_feature.reshape(batch_size, seqlen, 1024)
        img_res = img_feature
        
        for _ in range(6):
            img_feature_self = self.self_attention(img_feature)
            img_feature = img_feature + img_feature_self
            img_feature = self.norm(img_feature)
            img_feature = self.ffn(img_feature)
        
        img_feature = self.self_attention(img_feature)
        fusion_feature = self.ffn(img_feature)
        
        for _ in range(5):
            img_feature = self.self_attention(img_feature)
            fusion_feature = self.ffn(img_feature)
        
        fusion_feature = fusion_feature + img_res
        fusion_feature = self.layer_norm(fusion_feature)
        
        res_fusion = fusion_feature
        
        fusion_feature = self.ffn(fusion_feature)
        
        fusion_feature = fusion_feature + res_fusion
        fusion_feature = self.layer_norm(fusion_feature)
        
        fusion_feature = fusion_feature + img_res
        
        fusion_feature = fusion_feature.reshape(-1, fusion_feature.size(-1))
        # ----------------------------------------------
        smpl_output = self.regressor(fusion_feature)
        
        return feature, smpl_output

class VIBE_rgbpc(nn.Module):
    def __init__(self,
                 n_layers=2,
                 hidden_size=1024,
                 add_linear=True,
                 bidirectional=False,
                 use_residual=True):
        super(VIBE_rgbpc, self).__init__()
        self.encoder = TemporalEncoder(
            n_layers=n_layers,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            add_linear=add_linear,
            use_residual=use_residual,
        )
        self.regressor = Regressor()
        self.cross_attention = nn.MultiheadAttention(embed_dim = 1024, num_heads = 4, dropout = 0.1)
        self.ffn = PositionwiseFeedForward()
        self.self_attention = nn.MultiheadAttention(embed_dim = 1024, num_heads = 4, dropout = 0.1)
        self.fc = nn.Linear(2048, 1024)
        self.norm = nn.LayerNorm(normalized_shape=1024)
        self.layer_norm = nn.LayerNorm(1024, eps=1e-6)
        self.trans_encoder1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model = 1024, nhead = 4),
            num_layers = 6)
        self.trans_encoder2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model = 1024, nhead = 4),
            num_layers = 12)

    def forward(self, input, pc_feature = None):
        batch_size, seqlen = input.shape[:2]
        feature = self.encoder(input)
        
        feature = feature.reshape(-1, feature.size(-1))
        img_feature = self.fc(feature)
        
        img_feature = img_feature.reshape(batch_size, seqlen, 1024)
        pc_res = pc_feature
        img_res = img_feature
            
        pc_feature = self.trans_encoder2(pc_feature)
        pc_feature,_ = self.self_attention(pc_feature, pc_feature, pc_feature)
        
        img_feature = self.trans_encoder1(img_feature)
        img_feature,_ = self.self_attention(img_feature, img_feature, img_feature)
        
        fusion_feature,_ = self.cross_attention(pc_feature, img_feature, img_feature)
        fusion_feature = self.ffn(fusion_feature)
        
        for _ in range(5):
            pc_feature,_ = self.self_attention(pc_feature, pc_feature, pc_feature)
            img_feature,_ = self.self_attention(img_feature, img_feature, img_feature)
            fusion_feature,_ = self.cross_attention(pc_feature, fusion_feature, img_feature)
            fusion_feature = self.ffn(fusion_feature)
        
        fusion_feature = fusion_feature + pc_res
        fusion_feature = self.layer_norm(fusion_feature)
        
        res_fusion = fusion_feature
        
        fusion_feature = self.ffn(fusion_feature)
        
        fusion_feature = fusion_feature + res_fusion
        fusion_feature = self.layer_norm(fusion_feature)
        
        fusion_feature = fusion_feature + pc_res + img_res
        fusion_feature = fusion_feature.reshape(-1, fusion_feature.size(-1))
        # -------------------------------------------------
        smpl_output = self.regressor(fusion_feature)
        return feature, smpl_output

class VIBE_eventpc(nn.Module):
    def __init__(self,
                 n_layers=2,
                 hidden_size=1024,
                 add_linear=True,
                 bidirectional=False,
                 use_residual=True):
        super(VIBE_eventpc, self).__init__()
        self.encoder = TemporalEncoder(
            n_layers=n_layers,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            add_linear=add_linear,
            use_residual=use_residual,
        )
        self.regressor = Regressor()
        self.cross_attention = nn.MultiheadAttention(embed_dim = 1024, num_heads = 4, dropout = 0.1)
        self.ffn = PositionwiseFeedForward()
        self.self_attention = nn.MultiheadAttention(embed_dim = 1024, num_heads = 4, dropout = 0.1)
        self.fc = nn.Linear(2048, 1024)
        self.norm = nn.LayerNorm(normalized_shape=1024)
        self.layer_norm = nn.LayerNorm(1024, eps=1e-6)
        self.trans_encoder1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model = 1024, nhead = 4),
            num_layers = 6)
        self.trans_encoder2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model = 1024, nhead = 4),
            num_layers = 12)

    def forward(self, input, pc_feature = None):
        batch_size, seqlen = input.shape[:2]
        feature = self.encoder(input)
        
        feature = feature.reshape(-1, feature.size(-1))
        img_feature = self.fc(feature)
        
        img_feature = img_feature.reshape(batch_size, seqlen, 1024)
        pc_res = pc_feature
        img_res = img_feature
            
        pc_feature = self.trans_encoder2(pc_feature)
        pc_feature,_ = self.self_attention(pc_feature, pc_feature, pc_feature)
        
        img_feature = self.trans_encoder1(img_feature)
        img_feature,_ = self.self_attention(img_feature, img_feature, img_feature)
        
        fusion_feature,_ = self.cross_attention(pc_feature, img_feature, img_feature)
        fusion_feature = self.ffn(fusion_feature)
        
        for _ in range(5):
            pc_feature,_ = self.self_attention(pc_feature, pc_feature, pc_feature)
            img_feature,_ = self.self_attention(img_feature, img_feature, img_feature)
            fusion_feature,_ = self.cross_attention(pc_feature, fusion_feature, img_feature)
            fusion_feature = self.ffn(fusion_feature)
        
        fusion_feature = fusion_feature + pc_res
        fusion_feature = self.layer_norm(fusion_feature)
        
        res_fusion = fusion_feature
        fusion_feature = self.ffn(fusion_feature)
        
        fusion_feature = fusion_feature + res_fusion
        fusion_feature = self.layer_norm(fusion_feature)
        
        fusion_feature = fusion_feature + pc_res + img_res
        
        fusion_feature = fusion_feature.reshape(-1, fusion_feature.size(-1))
        # -----------------------------------------------
        smpl_output = self.regressor(fusion_feature)
      
        return feature, smpl_output


if __name__ == '__main__':
    vibe = VIBE().cuda()
    x = torch.rand((6, 16, 2048)).cuda()
    x1 = torch.rand((6, 16, 1024)).cuda()
    output_1, _= vibe(x, x1)
    print(output_1.shape)
    for k in _.keys():
        print(k, _[k].shape)
