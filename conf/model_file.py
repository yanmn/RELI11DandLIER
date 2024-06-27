import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from os.path import join

RELI_DIR = "/data/RELI11D" # the path of the dataset

VIBE_DATA_DIR = "/data/vibe_data/vibe_wights"
SMPL_MEAN_PARAMS = join(VIBE_DATA_DIR, 'smpl_mean_params.npz')

ROOT_DATASET = '/data'

PEDX_ROOT = os.path.join(ROOT_DATASET, 'pedx')
SMPL_FILE = os.path.join(ROOT_DATASET, '/smpl', 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')

JOINT_REGRESSOR_TRAIN_EXTRA = os.path.join(ROOT_DATASET, 'lidarcap/smpl', 'J_regressor_extra.npy')




