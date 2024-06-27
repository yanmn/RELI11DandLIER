import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from dataset.reli import RELI


def get_data_loader(cfg):
    train_loader = DataLoader(
        dataset = RELI(dataset = "MMSD_train3.pkl", is_train = True),
        batch_size = cfg['train_bs'],
        num_workers = cfg['threads'],
        shuffle = True,
        drop_last = False
    )
    return train_loader