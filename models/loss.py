import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from models.smpl import SMPL
from utils.geometry import batch_rodrigues

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.criterion_pose = nn.MSELoss()
        self.criterion_joints = nn.MSELoss()
        self.criterion_vertices = nn.MSELoss()

        self.criterion_keypoints = nn.MSELoss()
        self.criterion_regr = nn.MSELoss()

        self.smpl = SMPL().cuda()

    def forward(self, preds, gt):
        B, T = gt['point'].shape[:2]

        loss_joints = self.criterion_joints(preds['joint'], gt["joint_3d_cam"])
        
        gt_pose = gt['pose']
        gt_rotmats = batch_rodrigues(gt_pose.reshape(-1, 3)).reshape(B, T, 24, 3, 3)
        loss_pose = self.criterion_pose(preds['rotmat'], gt_rotmats)
        
        pred_human_vertices = self.smpl(
            pose = preds['rotmat'].reshape(-1, 24, 3, 3),
            beta = torch.zeros((B * T, 10)).cuda()
        )
        
        pred_smpl_joints = self.smpl.get_full_joints(
            vertices = pred_human_vertices
        ).reshape(B, T, 24, 3)

        loss_smpl_joints = self.criterion_joints(pred_smpl_joints, gt["joint_3d_pc"])
     
        loss_kp_3d = self.keypoint_3d_loss(preds['img_kp_3d'], gt['joint_3d_cam'])
        
        loss_kp_2d = self.criterion_keypoints(preds['img_kp_2d'], gt['joint_2d'])
        
        pred_rotmats = batch_rodrigues(preds['img_theta'][:, :, 3:75].reshape(-1, 3)).reshape(B, T, 24, 3, 3)
        loss_pose_2d = self.criterion_regr(pred_rotmats, gt_rotmats) # MSELoss
        loss_shape_2d = self.criterion_regr(preds['img_theta'][:, :, 75:], gt['shape']) 

        loss_dict = {
            'loss_joint': loss_joints,
            'loss_pose': loss_pose,
            'loss_smpl_joint': loss_smpl_joints,
            'loss_kp_3d': loss_kp_3d,
            'loss_kp_2d': loss_kp_2d,
            'loss_pose_2d': loss_pose_2d,
            'loss_shape_2d': loss_shape_2d
        }
        total_loss = torch.stack(list(loss_dict.values())).sum()
        loss_dict['total_loss'] = total_loss

        return loss_dict

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d):
        gt_keypoints_3d = gt_keypoints_3d - gt_keypoints_3d[:, 0:1, :]
        pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, 0:1, :]
        
        return self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)

