import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from conf.model_file import RELI_DIR # the path of the dataset
from models.regressor import Regressor_all, Regressor_rgb, Regressor_pc, Regressor_event, Regressor_rgbpc, Regressor_eventpc
from dataset.reli_vibe import RELI
from utils.utils import move_dict_to_device
from utils.geometry import  get_pred_poses
from utils.metric_utils import output_metric
import pickle

def load_pretrained(model, model_path):
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["state_dict"])
        print("load pretrained model successfully!")
    else:
        print("model_path is none!")
        exit()
    
def compute_errors_joint(gt3ds, preds):
    """
    Gets MPJPE after pelvis alignment + MPJPE after Procrustes.
    Evaluates on the 24 common joints.
    Inputs:
      - gt3ds: N x 24 x 3
      - preds: N x 24 x 3
    """
    errors = []
    errors = np.sqrt(((preds - gt3ds) ** 2).sum(-1)).mean(-1)

    return errors
        
def mean(a, l):
    x = 0
    for i in range(len(a)):
        x += a[i] * l[i]
    return x / sum(l)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type = str, default = '0', help = '')
    parser.add_argument('--ckpt_path', type = str,
                        default = '',
                        help = 'the .pth file path of pretrained model')
    parser.add_argument('--test', type = str,
                        default = '', help = 'test data')
    parser.add_argument('--file_name', type = str, default = '',help = 'the path of the saved results')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.makedirs(args.file_name, exist_ok = True)
    results_txt = os.path.join(args.file_name, 'results.txt')
    eval_info = open(results_txt, mode="a", encoding="utf-8")

    regressor = Regressor_all().cuda() # which could be replaced by Regressor_rgb, Regressor_rgbpc, ... 
    load_pretrained(regressor, args.ckpt_path)
    regressor = regressor.eval()

    test_list = ["test_zy3.pkl", "test_zmh3.pkl", "test_ym3.pkl", "test_qzk3.pkl", "test_kys3.pkl", "test_csq3.pkl"]
    # the list of test sets
    # test_zy3.pkl, test_zmh3.pkl, test_ym3.pkl, ...
    
    len_list = []
    indices_dict = {
        "accel_error" : [],
        "mpjpe": [],
        "pa_mpjpe": [],
        "pve": [],
        "pck_30": [],
        "pck_50": [],
        "trans_error" : [],
        "trans_error_fix" : [],
        "g_mpjpe" : [],
    }
    for test_data in test_list:
        eval_loader = DataLoader(dataset = RELI(dataset = test_data, is_eval = True),
                                batch_size = 6,
                                num_workers = 2)
        eval_iter = iter(eval_loader)
        bar = tqdm(eval_iter, bar_format="{l_bar}{bar:3}{r_bar}", ncols = 110)
        bar.set_description(f'Running ------>')
        rotmats = np.zeros((0, 16, 24, 3, 3))

        gt_poses = []
        gt_trans = [] 
        gt_joint3d = []
        pred_joint3d = []
        pred_trans = []
        pred_shape = []
        
        for _, inputs in enumerate(bar):
            move_dict_to_device(dict=inputs, device="cuda")
            gt_poses.append(inputs["pose"].cpu().detach().numpy())
            gt_trans.append(inputs['trans'].cpu().detach().numpy()) 
            gt_joint3d.append(inputs["joint_3d_cam"].cpu().detach().numpy())
            with torch.no_grad():
                preds = regressor(inputs)
                pred_trans.append(preds['joint'][:,:,0,:].cpu().detach().numpy()) 
                pred_shape.append(preds['img_theta'][:,:,-10:].cpu().detach().numpy()) 
                pred_rotmats = preds["rotmat"]
                pred_joint3d.append(preds["joint"].cpu().detach().numpy())
                rotmats = np.concatenate((rotmats, pred_rotmats.cpu().detach().numpy()), axis=0)

        gt_poses = np.concatenate(gt_poses, axis=0).reshape(-1, 72)
        gt_joint3d = np.concatenate(gt_joint3d, axis=0).reshape(-1, 16, 24, 3)
        gt_trans = np.concatenate(gt_trans, axis=0).reshape(-1, 3)
        pred_joint3d = np.concatenate(pred_joint3d, axis=0).reshape(-1, 16, 24, 3)
        pred_poses = get_pred_poses(rotmats.reshape(-1, 24, 3, 3))
        pred_trans = np.concatenate(pred_trans, axis=0).reshape(-1, 3)
        pred_shape = np.concatenate(pred_shape, axis=0).reshape(-1, 10)
        
        test_filename = os.path.join(RELI_DIR, test_data)
        save_filename = os.path.join(args.file_name, test_data[:-4] + "_pred.npz")
        with open(test_filename, "rb") as file:
            data = pickle.load(file)

        np.savez(save_filename,
                pred_pose = pred_poses,
                points_cloud = data["point_clouds_2D"].reshape(-1, 512, 3),
                pose = gt_poses,
                pred_trans = pred_trans,
                trans = gt_trans,
                joint = pred_joint3d,
                pred_shape = pred_shape,
                )
        len_list.append(gt_poses.shape[0])
        accel_error, mpjpe, pa_mpjpe, pve, pck_30, pck_50 = output_metric(pred_poses=pred_poses, gt_poses=gt_poses)

        trans_error_fix = pred_trans[0] - gt_trans[0]
        pred_trans_fix = pred_trans - trans_error_fix
        
        err1 = []
        for i in range(pred_trans.shape[0]):
            err1.append(np.linalg.norm(pred_trans[i] - gt_trans[i])) 
 
        sum_err1 = sum(err1)
        sum_err1 = sum_err1 / pred_trans.shape[0]
        
        err2 = []

        for i in range(pred_trans.shape[0]):
            err2.append(np.linalg.norm(pred_trans_fix[i] - gt_trans[i])) 

        sum_err2 = sum(err2)
        sum_err2 = sum_err2 / pred_trans.shape[0]
        
        g_mpjpe = compute_errors_joint(pred_joint3d, gt_joint3d)
  
        m2mm = 1000
        g_mpjpe = np.mean(g_mpjpe) * m2mm

        indices_dict["accel_error"].append(accel_error)
        indices_dict["mpjpe"].append(mpjpe)
        indices_dict["pa_mpjpe"].append(pa_mpjpe)
        indices_dict["pve"].append(pve)
        indices_dict["pck_30"].append(pck_30)
        indices_dict["pck_50"].append(pck_50)
        indices_dict["trans_error"].append(sum_err1) 
        indices_dict["trans_error_fix"].append(sum_err2) 
        indices_dict["g_mpjpe"].append(g_mpjpe) 

        print(f'-------------{test_data}------------', file = eval_info)
        print('accel_error : {:.6f}'.format(accel_error), file = eval_info)
        print('mpjpe       : {:.6f}'.format(mpjpe), file = eval_info)
        print('pa_mpjpe    : {:.6f}'.format(pa_mpjpe), file = eval_info)
        print('g_mpjpe    : {:.6f}'.format(g_mpjpe), file = eval_info) 
        print('pve         : {:.6f}'.format(pve), file = eval_info)
        print('pck_30      : {:.6f}'.format(pck_30), file = eval_info)
        print('pck_50      : {:.6f}'.format(pck_50), file = eval_info)
        print('trans_error : ', file = eval_info) 
        print(sum_err1, file = eval_info) 
        print('trans_error_fix : ', file = eval_info) 
        print(sum_err2, file = eval_info) 
        print('num_frames  : {:.6f}'.format(gt_poses.shape[0]), file = eval_info)
        print("-----------------------------------------------")

    accel_error, mpjpe, pa_mpjpe, pve, pck_30, pck_50, trans_error, trans_error_fix, g_mpjpe = 0, 0, 0, 0, 0, 0, 0, 0, 0
 
    for i, l in enumerate(len_list):
        accel_error += indices_dict["accel_error"][i] * l
        mpjpe += indices_dict["mpjpe"][i] * l
        pa_mpjpe += indices_dict["pa_mpjpe"][i] * l
        g_mpjpe += indices_dict["g_mpjpe"][i] * l 
        pve += indices_dict["pve"][i] * l
        pck_30 += indices_dict["pck_30"][i] * l
        pck_50 += indices_dict["pck_50"][i] * l
        trans_error += indices_dict["trans_error"][i] * l 
        trans_error_fix += indices_dict["trans_error_fix"][i] * l 
        
    accel_error /= sum(len_list)
    mpjpe /= sum(len_list)
    pa_mpjpe /= sum(len_list)
    g_mpjpe /= sum(len_list) 
    pve /= sum(len_list)
    pck_30 /= sum(len_list)
    pck_50 /= sum(len_list)
    trans_error /= sum(len_list) 
    trans_error_fix /= sum(len_list) 
    
    print('------------The final results-------------', file = eval_info)
    print('accel_error : {:.6f}'.format(accel_error), file = eval_info)
    print('mpjpe       : {:.6f}'.format(mpjpe), file = eval_info)
    print('pa_mpjpe    : {:.6f}'.format(pa_mpjpe), file = eval_info)
    print('g_mpjpe    : {:.6f}'.format(g_mpjpe), file = eval_info) 
    print('pve         : {:.6f}'.format(pve), file = eval_info)
    print('pck_30      : {:.6f}'.format(pck_30), file = eval_info)
    print('pck_50      : {:.6f}'.format(pck_50), file = eval_info)
    print('trans_error : ', file = eval_info) 
    print(trans_error, file = eval_info) 
    print('trans_error_fix : ', file = eval_info) 
    print(trans_error_fix, file = eval_info) 
    print()

    eval_info.close()

