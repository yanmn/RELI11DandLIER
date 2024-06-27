import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.stgcn import STGCN
from models.pointnet2 import PointNet2Encoder
from models.vibe import VIBE, VIBE2, VIBE_rgb, VIBE_pc, VIBE_event, VIBE_rgbpc, VIBE_eventpc
from models.cross_attention import PositionwiseFeedForward
from models.self_attention import SelfAttention
from utils.geometry import rot6d_to_rotmat

import pickle as pkl

class RNN(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=2):
        super(RNN, self).__init__()
        self.rnn = nn.GRU(
            n_hidden,
            n_hidden,
            n_rnn_layer,
            batch_first=True,
            bidirectional=True
        )
        self.linear1 = nn.Linear(n_input, n_hidden)
        self.linear2 = nn.Linear(n_hidden * 2, n_output)

    def forward(self, x):
        x = self.rnn(F.relu(F.dropout(self.linear1(x)), inplace=True))[0]
        return self.linear2(x)


class Regressor_rgb(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PointNet2Encoder()
        self.rnn = RNN(1024, 24 * 3, 1024)
        self.stgcn = STGCN(3 + 1024)
        self.vibe = VIBE_rgb(n_layers=2,
                         hidden_size=1024,
                         add_linear=True,
                         bidirectional=False,
                         use_residual=True)
        self.ffn = PositionwiseFeedForward()
        self.self_attention = SelfAttention()
        self.fc = nn.Linear(2048, 1024)
        self.norm = nn.LayerNorm(normalized_shape=1024)
        self.layer_norm = nn.LayerNorm(1024, eps=1e-6)

    def forward(self, data):
        # print("Using RGB figures......")
        pred = {}
        B, T = data["img_feature"].shape[:2]
        D = 1024

        img_feature, smpl_output = self.vibe(data["img_feature"])
        img_feature_detached = self.fc(img_feature).view(B, T, D).detach()
     
        img_res = img_feature_detached
        img_feature = img_feature_detached

        for _ in range(6):
            img_feature_self = self.self_attention(img_feature)
            img_feature = img_feature_self + img_feature
            img_feature = self.norm(img_feature)
            img_feature = self.ffn(img_feature)
        
        img_feature = self.self_attention(img_feature)
        feature = self.ffn(img_feature)
        
        for _ in range(5):
            img_feature = self.self_attention(img_feature)
            feature = self.ffn(img_feature)
        
        feature = feature + img_res
        feature = self.layer_norm(feature)
        
        res_fusion = feature
        
        feature = self.ffn(feature)
        
        feature = feature + res_fusion
        feature = self.layer_norm(feature)
        
        feature = feature + img_res
        # ------------------------------------------------------------------------
        B, T, _ = feature.shape
        joint = self.rnn(feature)
        rot6ds = self.stgcn(
            torch.cat(
                (
                    joint.reshape(B, T, 24, 3),
                    feature.unsqueeze(-2).repeat(1, 1, 24, 1)
                ),
                dim=-1
            )
        )
        
        rot6ds = rot6ds.reshape(-1, rot6ds.size(-1))
        rotmats = rot6d_to_rotmat(rot6ds).reshape(-1, 3, 3)
        
        pred['rotmat'] = rotmats.reshape(B, T, 24, 3, 3)
        pred['joint'] = joint.reshape(B, T, 24, 3)
        pred['img_theta'] = smpl_output['theta'].reshape(B, T, 85)
        pred['img_kp_2d'] = smpl_output['kp_2d'].reshape(B, T, 24, 2)
        pred['img_kp_3d'] = smpl_output['kp_3d'].reshape(B, T, 24, 3)
        pred['img_rotmat'] = smpl_output['rotmat'].reshape(B, T, 24, 3, 3)
            
        return pred

class Regressor_pc(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PointNet2Encoder()
        self.rnn = RNN(1024, 24 * 3, 1024)
        self.stgcn = STGCN(3 + 1024)
        self.vibe = VIBE_pc(n_layers=2,
                         hidden_size=1024,
                         add_linear=True,
                         bidirectional=False,
                         use_residual=True)
        self.ffn = PositionwiseFeedForward()
        self.self_attention = SelfAttention()
        self.fc = nn.Linear(2048, 1024)
        self.norm = nn.LayerNorm(normalized_shape=1024)
        self.layer_norm = nn.LayerNorm(1024, eps=1e-6)

    def forward(self, data):
        # print("Using poind clouds......")

        pred = {}
        # ----------------------------------------------------------------------------
        # Choose one of the following options:
        # Option 1: Use the feature of point clouds directly
        # Use the line below and delete the line `pc_feature = data["pc_feature"]`
        pc_feature = self.encoder(data["point"])
        save_pc_f = pc_feature
        


        B, T, D = pc_feature.shape

        smpl_output = self.vibe(pc_feature = pc_feature.detach())
        pc_res = pc_feature
        
        for _ in range(6):
            pc_feature_self = self.self_attention(pc_feature)
            pc_feature = pc_feature + pc_feature_self
            pc_feature = self.norm(pc_feature)
            pc_feature = self.ffn(pc_feature)
            
        pc_feature = self.self_attention(pc_feature)
        feature = self.ffn(pc_feature)
        
        for _ in range(5):
            pc_feature = self.self_attention(pc_feature)
            feature = self.ffn(feature)

        feature = feature + pc_res
        feature = self.layer_norm(feature)
        
        res_fusion = feature
        
        feature = self.ffn(feature)
        
        feature = feature + res_fusion
        feature = self.layer_norm(feature)
        
        feature = feature + pc_res
        
        # ------------------------------------------------------------------
        B, T, _ = feature.shape
        joint = self.rnn(feature)
        rot6ds = self.stgcn(
            torch.cat(
                (
                    joint.reshape(B, T, 24, 3),
                    feature.unsqueeze(-2).repeat(1, 1, 24, 1)
                ),
                dim=-1
            )
        )
        rot6ds = rot6ds.reshape(-1, rot6ds.size(-1))
        rotmats = rot6d_to_rotmat(rot6ds).reshape(-1, 3, 3)
        
        pred['rotmat'] = rotmats.reshape(B, T, 24, 3, 3)
        pred['joint'] = joint.reshape(B, T, 24, 3)
        pred['img_theta'] = smpl_output['theta'].reshape(B, T, 85)
        pred['img_kp_2d'] = smpl_output['kp_2d'].reshape(B, T, 24, 2)
        pred['img_kp_3d'] = smpl_output['kp_3d'].reshape(B, T, 24, 3)
        pred['img_rotmat'] = smpl_output['rotmat'].reshape(B, T, 24, 3, 3)
            
        return pred, save_pc_f

class Regressor_event(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PointNet2Encoder()
        self.rnn = RNN(1024, 24 * 3, 1024)
        self.stgcn = STGCN(3 + 1024)
        self.vibe = VIBE_event(n_layers=2,
                         hidden_size=1024,
                         add_linear=True,
                         bidirectional=False,
                         use_residual=True)
        self.ffn = PositionwiseFeedForward()
        self.self_attention = SelfAttention()
        self.fc = nn.Linear(2048, 1024)
        self.norm = nn.LayerNorm(normalized_shape=1024)
        self.layer_norm = nn.LayerNorm(1024, eps=1e-6)

    def forward(self, data):
        # print("Using event streams......")
        pred = {}
        B, T = data["event_feature"].shape[:2]
        D = 1024

        event_feature, smpl_output = self.vibe(data["event_feature"])
        event_feature_detached = self.fc(event_feature).view(B, T, D).detach()
        
        event_res = event_feature_detached
        event_feature = event_feature_detached

        for _ in range(6):
            event_feature_self = self.self_attention(event_feature)
            event_feature = event_feature_self + event_feature
            event_feature = self.norm(event_feature)
            event_feature = self.ffn(event_feature)
        
        event_feature = self.self_attention(event_feature)
        feature = self.ffn(event_feature)
        
        for _ in range(5):
            event_feature = self.self_attention(event_feature)
            feature = self.ffn(event_feature)
        
        feature = feature + event_res
        feature = self.layer_norm(feature)
        
        res_fusion = feature
        
        feature = self.ffn(feature)
        
        feature = feature + res_fusion
        feature = self.layer_norm(feature)
        
        feature = feature + event_res
        # ------------------------------------------------------------------
        B, T, _ = feature.shape
        joint = self.rnn(feature)
        rot6ds = self.stgcn(
            torch.cat(
                (
                    joint.reshape(B, T, 24, 3),
                    feature.unsqueeze(-2).repeat(1, 1, 24, 1)
                ),
                dim=-1
            )
        )
        rot6ds = rot6ds.reshape(-1, rot6ds.size(-1))
        rotmats = rot6d_to_rotmat(rot6ds).reshape(-1, 3, 3)
        
        pred['rotmat'] = rotmats.reshape(B, T, 24, 3, 3)
        pred['joint'] = joint.reshape(B, T, 24, 3)
        pred['img_theta'] = smpl_output['theta'].reshape(B, T, 85)
        pred['img_kp_2d'] = smpl_output['kp_2d'].reshape(B, T, 24, 2)
        pred['img_kp_3d'] = smpl_output['kp_3d'].reshape(B, T, 24, 3)
        pred['img_rotmat'] = smpl_output['rotmat'].reshape(B, T, 24, 3, 3)
            
        return pred

class Regressor_rgbpc(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PointNet2Encoder()
        self.rnn = RNN(1024, 24 * 3, 1024)
        self.stgcn = STGCN(3 + 1024)
        self.vibe = VIBE_rgbpc(n_layers=2,
                         hidden_size=1024,
                         add_linear=True,
                         bidirectional=False,
                         use_residual=True)
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

    def forward(self, data):
        # print("Using RGB figures and poind clouds......")
        pred = {}
        # ----------------------------------------------------------------------------
        # Choose one of the following options:
        # Option 1: Use the feature of point clouds directly
        # Use the line below and delete the line `pc_feature = data["pc_feature"]`
        pc_feature = self.encoder(data["point"])

        # Option 2: Extract features here
        # Use the line `pc_feature = data["pc_feature"]` and delete the line above
        pc_feature = data["pc_feature"]
        # -----------------------------------------------------------------------------
        B, T, D = pc_feature.shape

        img_feature, smpl_output = self.vibe(data["img_feature"], pc_feature = pc_feature.detach())
        img_feature_detached = self.fc(img_feature).view(B, T, D).detach()
        
        pc_res = pc_feature
        img_res = img_feature_detached
        img_feature = img_feature_detached
        
        pc_feature = self.trans_encoder2(pc_feature)
        pc_feature,_ = self.self_attention(pc_feature, pc_feature, pc_feature)
        
        img_feature = self.trans_encoder1(img_feature)
        img_feature,_ = self.self_attention(img_feature, img_feature, img_feature)
        
        feature,_ = self.cross_attention(pc_feature, img_feature, img_feature)
        feature = self.ffn(feature)
        
        for _ in range(5):
            pc_feature,_ = self.self_attention(pc_feature, pc_feature, pc_feature)
            img_feature,_ = self.self_attention(img_feature, img_feature, img_feature)
            feature,_ = self.cross_attention(pc_feature, feature, img_feature)
            feature = self.ffn(feature)
        
        feature = feature + pc_res
        feature = self.layer_norm(feature)
        
        res_fusion = feature
        
        feature = self.ffn(feature)
        
        feature = feature + res_fusion
        feature = self.layer_norm(feature)
        
        feature = feature + pc_res + img_res
        # ---------------------------------------------------------------------------
        B, T, _ = feature.shape
        joint = self.rnn(feature)
        rot6ds = self.stgcn(
            torch.cat(
                (
                    joint.reshape(B, T, 24, 3),
                    feature.unsqueeze(-2).repeat(1, 1, 24, 1)
                ),
                dim=-1
            )
        )
        
        rot6ds = rot6ds.reshape(-1, rot6ds.size(-1))
        rotmats = rot6d_to_rotmat(rot6ds).reshape(-1, 3, 3)
        
        pred['rotmat'] = rotmats.reshape(B, T, 24, 3, 3)
        pred['joint'] = joint.reshape(B, T, 24, 3)
        pred['img_theta'] = smpl_output['theta'].reshape(B, T, 85)
        pred['img_kp_2d'] = smpl_output['kp_2d'].reshape(B, T, 24, 2)
        pred['img_kp_3d'] = smpl_output['kp_3d'].reshape(B, T, 24, 3)
        pred['img_rotmat'] = smpl_output['rotmat'].reshape(B, T, 24, 3, 3)
            
        return pred

class Regressor_eventpc(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PointNet2Encoder()
        self.rnn = RNN(1024, 24 * 3, 1024)
        self.stgcn = STGCN(3 + 1024)
        self.vibe = VIBE_eventpc(n_layers=2,
                         hidden_size=1024,
                         add_linear=True,
                         bidirectional=False,
                         use_residual=True)
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

    def forward(self, data):
        # print("Using event streams and point clouds......")
        pred = {}
        # ----------------------------------------------------------------------------
        # Choose one of the following options:
        # Option 1: Use the feature of point clouds directly
        # Use the line below and delete the line `pc_feature = data["pc_feature"]`
        pc_feature = self.encoder(data["point"])

        # Option 2: Extract features here
        # Use the line `pc_feature = data["pc_feature"]` and delete the line above
        pc_feature = data["pc_feature"]
        # -----------------------------------------------------------------------------
        
        B, T, D = pc_feature.shape
   
        event_feature, smpl_output = self.vibe(data["event_feature"], pc_feature = pc_feature.detach())
        event_feature_detached = self.fc(event_feature).view(B, T, D).detach()
        
        pc_res = pc_feature
        event_res = event_feature_detached
        event_feature = event_feature_detached
        
        pc_feature = self.trans_encoder2(pc_feature)
        pc_feature,_ = self.self_attention(pc_feature, pc_feature, pc_feature)
        
        event_feature = self.trans_encoder1(event_feature)
        event_feature,_ = self.self_attention(event_feature, event_feature, event_feature)
        
        feature,_ = self.cross_attention(pc_feature, event_feature, event_feature)
        feature = self.ffn(feature)
        
        for _ in range(5):
            pc_feature,_ = self.self_attention(pc_feature, pc_feature, pc_feature)
            event_feature,_ = self.self_attention(event_feature, event_feature, event_feature)
            feature,_ = self.cross_attention(pc_feature, feature, event_feature)
            feature = self.ffn(feature)
        
        feature = feature + pc_res
        feature = self.layer_norm(feature)
        
        res_fusion = feature
        feature = self.ffn(feature)
        
        feature = feature + res_fusion
        feature = self.layer_norm(feature)
        
        feature = feature + pc_res + event_res
        # -------------------------------------------------------------------------
        B, T, _ = feature.shape
        joint = self.rnn(feature)
        rot6ds = self.stgcn(
            torch.cat(
                (
                    joint.reshape(B, T, 24, 3),
                    feature.unsqueeze(-2).repeat(1, 1, 24, 1)
                ),
                dim=-1
            )
        )
        rot6ds = rot6ds.reshape(-1, rot6ds.size(-1))
        rotmats = rot6d_to_rotmat(rot6ds).reshape(-1, 3, 3)
        
        pred['rotmat'] = rotmats.reshape(B, T, 24, 3, 3)
        pred['joint'] = joint.reshape(B, T, 24, 3)
        pred['img_theta'] = smpl_output['theta'].reshape(B, T, 85)
        pred['img_kp_2d'] = smpl_output['kp_2d'].reshape(B, T, 24, 2)
        pred['img_kp_3d'] = smpl_output['kp_3d'].reshape(B, T, 24, 3)
        pred['img_rotmat'] = smpl_output['rotmat'].reshape(B, T, 24, 3, 3)
            
        return pred

class Regressor_all(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PointNet2Encoder()
        self.rnn = RNN(1024, 24 * 3, 1024)
        self.stgcn = STGCN(3 + 1024)
        self.vibe = VIBE(n_layers=2,
                         hidden_size=1024,
                         add_linear=True,
                         bidirectional=False,
                         use_residual=True)
        self.vibe2 = VIBE2(n_layers=2,
                         hidden_size=1024,
                         add_linear=True,
                         bidirectional=False,
                         use_residual=True)
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

    def forward(self, data):
        # print("Using all the three modalities......")
        pred = {}
        # ----------------------------------------------------------------------------
        # Choose one of the following options:
        # Option 1: Use the feature of point clouds directly
        # Use the line below and delete the line `pc_feature = data["pc_feature"]`
        pc_feature = self.encoder(data["point"])

        # Option 2: Extract features here
        # Use the line `pc_feature = data["pc_feature"]` and delete the line above
        pc_feature = data["pc_feature"]
        # ----------------------------------------------------------------------------- 
        B, T, D = pc_feature.shape
        
        img_feature, smpl_output = self.vibe(data["img_feature"], pc_feature = pc_feature.detach())

        img_feature_detached = self.fc(img_feature).view(B, T, D).detach()

        pc_res = pc_feature
        img_res = img_feature_detached
        img_feature = img_feature_detached
        
        pc_feature = self.trans_encoder2(pc_feature)  
        pc_feature,_ = self.self_attention(pc_feature, pc_feature, pc_feature)
        
        img_feature = self.trans_encoder1(img_feature)
        img_feature,_ = self.self_attention(img_feature, img_feature, img_feature)
        
        feature,_ = self.cross_attention(pc_feature, img_feature, img_feature)
        feature = self.ffn(feature)
        
        for _ in range(5):
            pc_feature,_ = self.self_attention(pc_feature, pc_feature, pc_feature)
            img_feature,_ = self.self_attention(img_feature, img_feature, img_feature)
            feature,_ = self.cross_attention(pc_feature, feature, img_feature)
            feature = self.ffn(feature)

        feature = feature + pc_res
        feature = self.layer_norm(feature)
        
        res_fusion = feature
        
        feature = self.ffn(feature)
        
        feature = feature + res_fusion
        feature = self.layer_norm(feature)
        
        feature = feature + pc_res + img_res
        
        # rgb + pc
        # --------------------------------------------
        #  event + pc
        pc_feature = pc_res

        event_feature, smpl_output = self.vibe(data["event_feature"], pc_feature = pc_feature.detach())
        event_feature_detached = self.fc(event_feature).view(B, T, D).detach()
        
        pc_res = pc_feature
        event_res = event_feature_detached
        event_feature = event_feature_detached
        
        pc_feature = self.trans_encoder2(pc_feature)
        pc_feature,_ = self.self_attention(pc_feature, pc_feature, pc_feature)
        
        event_feature = self.trans_encoder1(event_feature)
        event_feature,_ = self.self_attention(event_feature, event_feature, event_feature)
        
        feature2,_ = self.cross_attention(pc_feature, event_feature, event_feature)
        feature2 = self.ffn(feature2)
        
        for _ in range(5):
            pc_feature,_ = self.self_attention(pc_feature, pc_feature, pc_feature)
            event_feature,_ = self.self_attention(event_feature, event_feature, event_feature)
            feature2,_ = self.cross_attention(pc_feature, feature2, event_feature)
            feature2 = self.ffn(feature2)
        
        feature2 = feature2 + pc_res
        feature2 = self.layer_norm(feature2)
        
        res_fusion = feature2
        
        feature2 = self.ffn(feature2)
        
        feature2 = feature2 + res_fusion
        feature2 = self.layer_norm(feature2)
        
        feature2 = feature2 + pc_res + event_res
        # ------------------------------------------------
        # fusion
        f1_res = feature
        f2_res = feature2
        
        _, smpl_output = self.vibe2(feature2, feature)
        
        feature = self.trans_encoder2(feature)
        feature,_ = self.self_attention(feature, feature, feature)
        
        feature2 = self.trans_encoder1(feature2)
        feature2,_ = self.self_attention(feature2, feature2, feature2)
        
        feature_all,_ = self.cross_attention(feature, feature2, feature2)
        feature_all = self.ffn(feature_all)
        
        for _ in range(5):
            feature,_ = self.self_attention(feature, feature, feature)
            feature2,_ = self.self_attention(feature2, feature2, feature2)
            feature_all,_ = self.cross_attention(feature, feature_all, feature2)
            feature_all = self.ffn(feature_all)
        
        feature_all = feature_all + f1_res
        feature_all = self.layer_norm(feature_all)
        
        res_fusion = feature_all
        
        feature_all = self.ffn(feature_all)
        
        feature_all = feature_all + res_fusion
        feature_all = self.layer_norm(feature_all)
        
        feature_all = feature_all + f1_res + f2_res
        
        # ---------------------------------------------------------------------------
        
        B, T, _ = feature_all.shape
        joint = self.rnn(feature_all)
        rot6ds = self.stgcn(
            torch.cat(
                (
                    joint.reshape(B, T, 24, 3),
                    feature_all.unsqueeze(-2).repeat(1, 1, 24, 1)
                ),
                dim=-1
            )
        )
        
        rot6ds = rot6ds.reshape(-1, rot6ds.size(-1))
        rotmats = rot6d_to_rotmat(rot6ds).reshape(-1, 3, 3)
        
        pred['rotmat'] = rotmats.reshape(B, T, 24, 3, 3)
        pred['joint'] = joint.reshape(B, T, 24, 3)
        pred['img_theta'] = smpl_output['theta'].reshape(B, T, 85)
        pred['img_kp_2d'] = smpl_output['kp_2d'].reshape(B, T, 24, 2)
        pred['img_kp_3d'] = smpl_output['kp_3d'].reshape(B, T, 24, 3)
        pred['img_rotmat'] = smpl_output['rotmat'].reshape(B, T, 24, 3, 3)
            
        return pred



if __name__ == "__main__":
    model = Regressor_all().cuda()
    point = torch.randn((1, 3, 1024, 3)).cuda()
    img_feature = torch.randn((1, 3, 2048)).cuda()
    event_feature = torch.randn((1, 3, 2048)).cuda()
    
    data = {
        'point': point, # (B, T, 1024)
        'img_feature': img_feature, # (B, T, 2048)
        'event_feature': event_feature, # (B, T, 2048)
    }
    model(data)