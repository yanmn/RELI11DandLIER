import os 
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import wandb
import argparse
import torch.optim as optim

from models.regressor import Regressor_rgb, Regressor_pc, Regressor_event, Regressor_rgbpc, Regressor_eventpc, Regressor_all
from models.loss import Loss
from trainer import Trainer
from dataset.loaders import get_data_loader
from utils.utils import load_configs, torch_set_gpu, make_reproducible

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
    "--modality_settings",
    type=str,
    default="all",
    choices=["rgb", "pc", "event", "rgb+pc", "event+pc", "all"],
    help=(
        "Select the used modalities. Options are: "
        "'all' (default) to use all available modalities, "
        "'rgb' to use only rgb modality, "
        "'pc' to use only pc modality, "
        "'event' to use event modality, "
        "'rgb+pc' to use rgb and pc modality, "
        "'event+pc' to use event and pc modality."
    )
)

    parser.add_argument( "--pretrained", type = str, default = None, help = "Using pretrained model or not, None or the path of the pretained model")
    args = parser.parse_args()
  
    unique_token = "{}".format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))

    config = load_configs(configs='conf/config.yaml')

    iscuda = torch_set_gpu(config['gpu'])
    make_reproducible(iscuda, seed = 7)

    model_dir = os.path.join('trained_model', unique_token)
    os.makedirs(model_dir, exist_ok=True)

    if args.modality_settings == "rgb":
        loaders = get_data_loader(config)
        regressor = Regressor_rgb().cuda()
        
        optimizer = optim.Adam(
            params = regressor.parameters(),
            lr = config['lr'], # 0.0001
            weight_decay = 1e-4
        )
        
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer = optimizer,
            factor = 0.9,
            patience = 1,
            threshold_mode = 'rel',
            threshold = 0.01,
            min_lr = 0.0000003
        )

        trainer = Trainer(
            config = config,
            loaders = loaders,
            criterion = Loss(),
            regressor = regressor,
            optimizer = optimizer,
            lr_scheduler = lr_scheduler,
            epochs = config['epochs'],
            device = 'cuda' if iscuda else 'cpu',
            model_dir = model_dir,
            eval_dir = unique_token,
            pre_trained = args.pretrained
        ).fit()

    elif args.modality_settings == "pc":
        loaders = get_data_loader(config)
        regressor = Regressor_pc().cuda()
        
        optimizer = optim.Adam(
            params = regressor.parameters(),
            lr = config['lr'], # 0.0001
            weight_decay = 1e-4
        )
        
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer = optimizer,
            factor = 0.9,
            patience = 1,
            threshold_mode = 'rel',
            threshold = 0.01,
            min_lr = 0.0000003
        )

        trainer = Trainer(
            config = config,
            loaders = loaders,
            criterion = Loss(),
            regressor = regressor,
            optimizer = optimizer,
            lr_scheduler = lr_scheduler,
            epochs = config['epochs'],
            device = 'cuda' if iscuda else 'cpu',
            model_dir = model_dir,
            eval_dir = unique_token,
            pre_trained = args.pretrained
        ).fit()

    elif args.modality_settings == "event":
        loaders = get_data_loader(config)
        regressor = Regressor_event().cuda()
        
        optimizer = optim.Adam(
            params = regressor.parameters(),
            lr = config['lr'], # 0.0001
            weight_decay = 1e-4
        )
        
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer = optimizer,
            factor = 0.9,
            patience = 1,
            threshold_mode = 'rel',
            threshold = 0.01,
            min_lr = 0.0000003
        )

        trainer = Trainer(
            config = config,
            loaders = loaders,
            criterion = Loss(),
            regressor = regressor,
            optimizer = optimizer,
            lr_scheduler = lr_scheduler,
            epochs = config['epochs'],
            device = 'cuda' if iscuda else 'cpu',
            model_dir = model_dir,
            eval_dir = unique_token,
            pre_trained = args.pretrained
        ).fit()

    elif args.modality_settings == "rgb+pc":
        loaders = get_data_loader(config)
        regressor = Regressor_rgbpc().cuda()
        
        optimizer = optim.Adam(
            params = regressor.parameters(),
            lr = config['lr'], # 0.0001
            weight_decay = 1e-4
        )
        
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer = optimizer,
            factor = 0.9,
            patience = 1,
            threshold_mode = 'rel',
            threshold = 0.01,
            min_lr = 0.0000003
        )

        trainer = Trainer(
            config = config,
            loaders = loaders,
            criterion = Loss(),
            regressor = regressor,
            optimizer = optimizer,
            lr_scheduler = lr_scheduler,
            epochs = config['epochs'],
            device = 'cuda' if iscuda else 'cpu',
            model_dir = model_dir,
            eval_dir = unique_token,
            pre_trained = args.pretrained
        ).fit()

    elif args.modality_settings == "event+pc":
        loaders = get_data_loader(config)
        regressor = Regressor_eventpc().cuda()
        
        optimizer = optim.Adam(
            params = regressor.parameters(),
            lr = config['lr'], # 0.0001
            weight_decay = 1e-4
        )
        
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer = optimizer,
            factor = 0.9,
            patience = 1,
            threshold_mode = 'rel',
            threshold = 0.01,
            min_lr = 0.0000003
        )

        trainer = Trainer(
            config = config,
            loaders = loaders,
            criterion = Loss(),
            regressor = regressor,
            optimizer = optimizer,
            lr_scheduler = lr_scheduler,
            epochs = config['epochs'],
            device = 'cuda' if iscuda else 'cpu',
            model_dir = model_dir,
            eval_dir = unique_token,
            pre_trained = args.pretrained
        ).fit()

    elif args.modality_settings == "all":
        loaders = get_data_loader(config)
        regressor = Regressor_all().cuda()
        
        optimizer = optim.Adam(
            params = regressor.parameters(),
            lr = config['lr'], # 0.0001
            weight_decay = 1e-4
        )
        
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer = optimizer,
            factor = 0.9,
            patience = 1,
            threshold_mode = 'rel',
            threshold = 0.01,
            min_lr = 0.0000003
        )

        trainer = Trainer(
            config = config,
            loaders = loaders,
            criterion = Loss(),
            regressor = regressor,
            optimizer = optimizer,
            lr_scheduler = lr_scheduler,
            epochs = config['epochs'],
            device = 'cuda' if iscuda else 'cpu',
            model_dir = model_dir,
            eval_dir = unique_token,
            pre_trained = args.pretrained
        ).fit()

    else:
        print("please choose the modalities...")
        exit()


if __name__ == '__main__':
    main()