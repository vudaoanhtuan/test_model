import os

import torch
import torch.nn as nn
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
import scipy.misc 



class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def update_step(self, values, step):
        """Log a scalar variable."""
        self.writer.add_scalars("step_loss", values, step)
        self.writer.flush()

    def update_epoch(self, values, step):
        self.writer.add_scalars("epoch_loss", values, step)
        self.writer.flush()


class Trainer:
    def __init__(self, model, optimizer, train_dl, test_dl, weight_dir='weight', log_dir='logs', scheduler=None, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_dl = train_dl
        self.test_dl = test_dl

        self.device = torch.device(device)
        self.model.to(self.device)

        self.weight_dir = weight_dir
        if not os.path.isdir(weight_dir):
            os.mkdir(weight_dir)

        if os.path.isdir(log_dir):
            import shutil
            shutil.rmtree(log_dir)

        self.logger = Logger(log_dir)
        self.train_step = 0


    def run_iterator(self, dataloader, is_training=True):
        if is_training:
            self.model.train()
        else:
            self.model.eval()
        
        total_loss = 0
        total_item = 0

        desc = "total_loss=%.6f | word_loss=%.6f | pos_loss=%.6f | lr=%.6f"
        with tqdm(total=len(dataloader)) as pbar:
            for src, mask, label, pos in dataloader:
                src = src.long().to(self.device)
                mask = mask.long().to(self.device)
                label = label.long().to(self.device)
                pos = pos.long().to(self.device)

                self.optimizer.zero_grad()
                output, cor_output, loss_word, loss_pos = self.model(src, mask, label, pos)
                loss = loss_word + loss_pos*1000

                if is_training:
                    loss.backward()
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()

                total_loss += loss.item()
                total_item += 1

                pbar.update(1)
                pbar.set_description(desc%(total_loss/total_item, loss_word.item(), loss_pos.item()*1000, self.optimizer.param_groups[0]['lr']))

                if is_training:
                    info = {"train_loss": loss.item()}
                    self.train_step += 1
                    step = self.train_step
                    self.logger.update_step(info, step)

        return total_loss/total_item

    def train(self, num_epoch=10):
        for epoch in range(num_epoch):
            print('\n[Epoch %d/%d] ========\n' % (epoch, num_epoch) ,flush=True, end='')
            train_loss = self.run_iterator(self.train_dl)
            torch.save(self.model.state_dict(), os.path.join(self.weight_dir, 'model.%02d.h5'%epoch))
            val_loss = self.run_iterator(self.test_dl, is_training=False)
            losses = {
                "train_loss": train_loss,
                "val_loss": val_loss
            }
            self.logger.update_epoch(losses, epoch)

