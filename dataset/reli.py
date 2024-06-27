import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from conf.model_file import RELI_DIR
from img_utils import transfrom_keypoints, normalize_2d_kp
import pickle

class RELI(Dataset):
    def __init__(
        self,
        dataset,
        is_train = False,
        is_val = False,
        is_eval = False
    ):
        super().__init__()

        self.dataset = dataset
        self.is_train = is_train
        self.is_val = is_val
        self.is_eval = is_eval
        self.data = self.load_pkl(self.dataset)
        self.img = self.data["img"]
        self.bbox = self.data['bbox']
        self.pc = self.data["point_clouds_2D"]

        self.pose = self.data["pose_2D"]
        self.shape = self.data["shape"]
        self.trans = self.data["trans_2D"]
        self.joint_3d_pc = self.data["joint3d_wo_trans_2D"]
        self.joint_3d_cam = self.data["joint3d_with_trans_2D"]
        self.joint_2d = self.data["joint2d"]
        self.img_feature = self.data["img_feature_ori"]
        self.event_feature = self.data["event_feature"]
        

    def load_npz(self, dataset):
        npz_file = os.path.join(RELI_DIR, dataset)
        if os.path.isfile(npz_file):
            npz = np.load(npz_file)
        else:
            raise ValueError(f"{self.dataset} do not exists!")
        print(f"Loaded {npz_file}!")
        return npz
    
    def load_pkl(self, dataset):
        pkl_file = os.path.join(RELI_DIR, dataset)
        with open(pkl_file, "rb") as file:
            pkl = pickle.load(file)
        print(f"Loaded {pkl_file}!")
        return pkl
    
    def pc_normalize(self, pc):
        pc[..., :] -= np.mean(pc[..., :], axis=1, keepdims=True)
        return pc

    def augment(self, pc, pc_num):
        T, N = pc.shape[:2]
        augment_pc = pc.copy()
        scale = np.random.uniform(0.9, 1.1)
        augment_pc *= scale
        dropout_ratio = np.clip(
            0,
            a_min=np.random.random()*(1-50/np.min(pc_num)),
            a_max=0.5
        )
        dropout_idx = np.where(np.random.random((T, N)) <= dropout_ratio)
        augment_pc[dropout_idx] = augment_pc[0][0]
        jittered_pc = np.clip(
            0.01*np.random.randn(*augment_pc.shape),
            a_min = -0.05,
            a_max = 0.05
        )
        augment_pc += jittered_pc
        return augment_pc

    def __getitem__(self, index):
        item = {}
        item["index"] = index

        point = self.pc[index].copy()
        item["point"] = torch.from_numpy(point).float()

        # item["pc_num"] = torch.from_numpy(self.pc_num[index]).float()   # with trans or not
        item["pose"] = torch.from_numpy(self.pose[index]).float()
        item["shape"] = torch.from_numpy(self.shape[index]).float()
        item["trans"] = torch.from_numpy(self.trans[index]).float()

        # ----------------------------------------------------------------- 
        joint2d = self.joint_2d[index]
        bbox = self.bbox[index]
        for i in range(joint2d.shape[0]):
            joint2d[i, :, :], trans = transfrom_keypoints(kp_2d=joint2d[i, :, :],
                                                   # x, y, w, h = bbox
                                                   center_x=bbox[i,0],
                                                   center_y=bbox[i,1],
                                                   width=bbox[i,2],
                                                   height=bbox[i,3],
                                                   patch_width=224,
                                                   patch_height=224,
                                                   do_augment=False)
            joint2d[i, :, :] = normalize_2d_kp(joint2d[i, :, :], 224)

        item["joint_2d"] = torch.from_numpy(joint2d).float()   
        # -----------------------------------------------------------------
        

        item["joint_3d_pc"] = torch.from_numpy(self.joint_3d_pc[index]).float()
        item["joint_3d_cam"] = torch.from_numpy(self.joint_3d_cam[index]).float()
        item["img_feature"] = torch.from_numpy(self.img_feature[index]).float()
        item["event_feature"] = torch.from_numpy(self.event_feature[index]).float()

  
        return item
    
    def __len__(self):
        return self.pose.shape[0]


if __name__ == "__main__":
    loader = DataLoader(
        dataset=RELI("train.npz", is_train=True),
        batch_size=6,
        num_workers=2
    )
    a = iter(loader)
    i = 0
    for j in a:
        if i <1:
            i+=1
            for k in j.keys():
                print(k, j[k].shape)
        else :
            break