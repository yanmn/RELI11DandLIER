import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import wandb
from tqdm import tqdm
from collections import defaultdict
from utils.utils import mean, move_dict_to_device

class Trainer():
    def __init__(self,
                 config,
                 loaders,
                 criterion,
                 regressor,
                 optimizer,
                 lr_scheduler = None,
                 epochs = 120,
                 device = 'cuda',
                 model_dir = '',
                 eval_dir = None,
                 pre_trained = None):
        self.train_loader = loaders
        self.criterion = criterion
        self.regressor = regressor
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        if pre_trained is not None:
            print('used!!!!!!!!!!!!!!')
            self.load_pretrained(pre_trained)
        self.config = config
        self.epochs = epochs
        self.device = device     
        self.model_dir = model_dir
        self.eval_dir = eval_dir
        self.log_step = 0
    
    def train(self, epoch):
        self.regressor.train()
        self.train_iter = iter(self.train_loader) 

        bar = tqdm(self.train_iter, bar_format="{l_bar}{bar:3}{r_bar}", ncols = 110)
        bar.set_description(f'Training {epoch:002d} ------>')

        loss_dict_detach = defaultdict(list)

        for _, inputs in enumerate(bar):
            move_dict_to_device(device = self.device, dict = inputs)
            preds = self.regressor(inputs)
            loss_dict = self.criterion(preds, inputs)

            self.optimizer.zero_grad()
            loss_dict["total_loss"].backward()
            self.optimizer.step()
                
            for k, v in loss_dict.items():
                if type(v) is not dict:
                    if isinstance(v, torch.Tensor):
                        loss_dict_detach[k].append(v.detach().cpu().numpy())
                    else:
                        loss_dict_detach[k].append(v)

            N = len(loss_dict_detach['total_loss']) // 10 + 1
            loss = loss_dict_detach['total_loss']
            bar.set_postfix(loss = f'{mean(loss[:N]):06.06f}--->'f'{mean(loss[-N:]):06.06f}'f'({mean(loss):06.06f})')

            del loss_dict, inputs

        final_loss = {'mean_' + k: sum(v) / len(v) for k, v in loss_dict_detach.items()}
        
        return final_loss


    def validate(self, epoch):
        self.regressor.eval()
        self.val_iter = iter(self.val_loader)

        bar = tqdm(self.val_iter, bar_format="{l_bar}{bar:3}{r_bar}", ncols = 110)
        bar.set_description(f'Validate {epoch:002d} ------>')

        loss_dict_detach = defaultdict(list)
        
        for _, inputs in enumerate(bar):
            move_dict_to_device(dict = inputs, device = self.device)
            with torch.no_grad():
                preds = self.regressor(inputs)
                loss_dict = self.criterion(preds, inputs)
                
            for k,v in loss_dict.items():
                if type(v) is not dict:
                    if isinstance(v, torch.Tensor):
                        loss_dict_detach[k].append(v.detach().cpu().numpy())
                    else:
                        loss_dict_detach[k].append(v)
            
            N = len(loss_dict_detach['total_loss']) // 10 + 1
            loss = loss_dict_detach['total_loss']
            bar.set_postfix(loss = f'{mean(loss[:N]):06.06f}--->'f'{mean(loss[-N:]):06.06f}'f'({mean(loss):06.06f})')
            del loss_dict, inputs
            
        final_loss = {'mean_' + k: sum(v) / len(v) for k, v in loss_dict_detach.items()}

        return final_loss


    def fit(self):
        train_logs_info_file = os.path.join(
            self.model_dir,
            'training_info.txt'
        )
        min_train_loss, best_train_acc = float('inf'), 0

        for epoch in range(1, self.epochs + 1):
            torch.cuda.empty_cache()

            train_loss_dict = self.train(epoch) 
            
            self.lr_scheduler.step(train_loss_dict['mean_loss_pose'])

            if train_loss_dict['mean_loss_pose'] < min_train_loss:
                min_train_loss = train_loss_dict['mean_loss_pose']
                
                self.save_model(
                    performance = min_train_loss,
                    epoch = epoch,
                    filename = os.path.join(self.model_dir, 'best_train_model.pth')
                )

                self.save_info(
                    epoch,
                    train_logs_info_file,
                )
            
            if epoch % 10 == 0 and epoch >= 170:
                self.save_model(
                    performance = train_loss_dict['mean_loss_pose'],
                    epoch = epoch,
                    filename = os.path.join(self.model_dir, f'train_model_{epoch}.pth')
                )

        print('Training and Validation done...\n')
    
    def save_model(self, performance, epoch, filename):
        save_dict = {
            'epoch': epoch,
            'performance': performance,
            'state_dict': self.regressor.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(save_dict, filename)

    def load_pretrained(self, model_path):
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)
            self.regressor.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            print("load pretrained model successfully!")
        else:
            print("model_path is none!")
            exit()

    def save_info(
        self,
        epoch,
        train_logs_info_file,
    ):

        train_logs_info = open(
            train_logs_info_file,
            mode="a",
            encoding="utf-8"
        )
        
        print('save .pth at {}'.format(epoch), file = train_logs_info)
        train_logs_info.close()
        