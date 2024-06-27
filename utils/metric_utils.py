
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from models.smpl import SMPL

def compute_accel(joints):
    """
    Computes acceleration of 3D joints.
    Args:
        joints (Nx25x3).
    Returns:
        Accelerations (N-2).
    """
    velocities = joints[1:] - joints[:-1]
    acceleration = velocities[1:] - velocities[:-1]
    acceleration_normed = np.linalg.norm(acceleration, axis=2)
    return np.mean(acceleration_normed, axis=1)


def compute_error_accel(joints_gt, joints_pred, vis=None):
    """
    Computes acceleration error:
        1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (Nx14x3).
        joints_pred (Nx14x3).
        vis (N).
    Returns:
        error_accel (N-2).
    """
    # (N-2)x14x3
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]
    
    normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    else:
        invis = np.logical_not(vis)
        invis1 = np.roll(invis, -1)
        invis2 = np.roll(invis, -2)
        new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
        new_vis = np.logical_not(new_invis)

    return np.mean(normed[new_vis], axis=1)


def poses_to_vertices(poses, trans=None):
    poses = poses.astype(np.float32)
    vertices = np.zeros((0, 6890, 3))

    n = len(poses)
    smpl = SMPL().cuda()

    batch_size = 128
    n_batch = (n + batch_size - 1) // batch_size

    for i in range(n_batch):
        lb = i * batch_size
        ub = (i + 1) * batch_size
        cur_n = min(ub - lb, n - lb)

        # Get vertices by SMPL
        cur_vertices = smpl(
            torch.from_numpy(poses[lb:ub]).cuda(),
            torch.zeros((cur_n, 10)).cuda()
        )
        
        vertices = np.concatenate((vertices, cur_vertices.cpu().numpy()))

    if trans is not None:
        trans = trans.astype(np.float32)
        vertices += np.expand_dims(trans, 1)

    return vertices


def poses_to_joints(poses):
    poses = poses.astype(np.float32)
    joints = np.zeros((0, 24, 3))

    n = len(poses)
    smpl = SMPL().cuda()

    batch_size = 128
    n_batch = (n + batch_size - 1) // batch_size

    for i in range(n_batch):
        lb = i * batch_size
        ub = (i + 1) * batch_size
        cur_n = min(ub - lb, n - lb)

        # Get vertices & joints by SMPL
        cur_vertices = smpl(
            torch.from_numpy(poses[lb:ub]).cuda(),
            torch.zeros((cur_n, 10)).cuda()
        )
        cur_joints = smpl.get_full_joints(cur_vertices)
        
        joints = np.concatenate((joints, cur_joints.cpu().numpy()))

    return joints


def compute_similarity_transform(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale * (R.dot(mu1))

    # 7. Error:
    S1_hat = scale * R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def compute_similarity_transform_torch(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert (S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1 ** 2)

    # 3. The outer product of X1 and X2.
    K = X1.mm(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)
    # V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[0], device=S1.device)
    Z[-1, -1] *= torch.sign(torch.det(U @ V.T))
    # Construct R.
    R = V.mm(Z.mm(U.T))

    # 5. Recover scale.
    scale = torch.trace(R.mm(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale * (R.mm(mu1))

    # 7. Error:
    S1_hat = scale * R.mm(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def batch_compute_similarity_transform_torch(S1, S2):
    '''
        Computes a similarity transform (sR, t) that takes
        a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
        where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
        i.e. solves the orthogonal Procrutes problem.

        S1: torch.from_numpy(preds).float() N x 24 x 3
        S2: torch.from_numpy(gt3ds).float() N x 24 x 3
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0, 2, 1)
        S2 = S2.permute(0, 2, 1)
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0, 2, 1)

    return S1_hat


def align_by_pelvis(joints):
    """
    Assumes joints is 14 x 3 in LSP order.
    Then hips are: [3, 2]
    Takes mid point of these points, then subtracts it.
    """
    pelvis = joints[0, :]
    return joints - np.expand_dims(pelvis, axis=0)


def compute_errors(gt3ds, preds):
    """
    Gets MPJPE after pelvis alignment + MPJPE after Procrustes.
    Evaluates on the 24 common joints.
    Inputs:
      - gt3ds: N x 24 x 3
      - preds: N x 24 x 3
    """
    gt3ds -= gt3ds[:, :1, :] 
    preds -= preds[:, :1, :] 
    errors, errors_pa = [], []
    errors = np.sqrt(((preds - gt3ds) ** 2).sum(-1)).mean(-1) 
    S1_hat = batch_compute_similarity_transform_torch(torch.from_numpy(preds).float(), torch.from_numpy(gt3ds).float()).numpy()
    errors_pa = np.sqrt(((S1_hat - gt3ds) ** 2).sum(-1)).mean(-1)
    
    return errors, errors_pa


def compute_error_verts(pred_verts, target_verts):
    """
    Computes MPJPE over 6890 surface vertices.
    Args:
        verts_gt (Nx6890x3).
        verts_pred (Nx6890x3).
    Returns:
        error_verts (N).
    """

    assert len(pred_verts) == len(target_verts)
    error_per_vert = np.sqrt(np.sum((target_verts - pred_verts) ** 2, axis=2))
    return np.mean(error_per_vert, axis=1)


torsal_length = 0.5127067


def compute_pck(pred_joints, gt_joints, threshold):
    B = len(pred_joints)
    pred_joints -= pred_joints[:, :1, :]
    gt_joints -= gt_joints[:, :1, :]

    distance = np.sqrt(((pred_joints - gt_joints) ** 2).sum(-1))
    correct = distance < threshold * torsal_length
    correct = (correct.sum(0) / B).mean()
    return correct


def compute_frame_pck(pred_joints, gt_joints, threshold):
    pred_joints -= pred_joints[:, :1, :]
    gt_joints -= gt_joints[:, :1, :]

    distance = np.sqrt(((pred_joints - gt_joints) ** 2).sum(-1))
    correct = distance < threshold * torsal_length
    correct = correct.mean(-1)
    return correct


def output_metric(pred_poses, gt_poses):
    pred_poses = pred_poses.astype(np.float32)
    gt_poses = gt_poses.astype(np.float32)

    pred_vertices   = poses_to_vertices(pred_poses)
    gt_vertices     = poses_to_vertices(gt_poses)
    
    pred_joints = poses_to_joints(pred_poses)
    gt_joints   = poses_to_joints(gt_poses)

    pred_joints     -= pred_joints[:, :1, :]
    gt_joints       -= gt_joints[:, :1, :]

    # The error metrics are measured in millimeters.
    m2mm = 1000 # m to mm

    # Acceleration error
    accel_error     = np.mean(compute_error_accel(gt_joints, pred_joints)) * m2mm 
    
    mpjpe, pa_mpjpe = compute_errors(gt_joints, pred_joints)

    mpjpe           = np.mean(mpjpe) * m2mm
    
    # Procrustes-Aligned Mean Per Joint Position Error (PA-MPJPE)
    pa_mpjpe        = np.mean(pa_mpjpe) * m2mm
    
    # Per Vertex Error (PVE)
    pve             = np.mean(compute_error_verts(pred_vertices, gt_vertices)) * m2mm
    
    # Percentage of Correct Keypoints (PCK)
    pck_30          = compute_pck(pred_joints, gt_joints, 0.3) # under 0.3m
    pck_50          = compute_pck(pred_joints, gt_joints, 0.5) # under 0.5m

    return accel_error, mpjpe, pa_mpjpe, pve, pck_30, pck_50
