# -*- coding = utf-8 -*-
# @File Name : train
# @Date : 2023/5/21 16:41
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import os
import json
import torch
import loss
import dataset
import network
import argparse
import torch.optim as optim

from tqdm import tqdm
from pathlib import Path
from logger import Logger
from datetime import date
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel


def read_json(config_file):
    """
    read the json file to config the training and dataset
    :param config_file: config file path
    :return: dictionary of config keys
    """
    config_file = Path(config_file)
    with config_file.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


class Trainer(object):
    def __init__(self, configer_file):
        # current date
        self.date = date.today()
        # get whole configurations
        self.config = read_json(configer_file)
        # dataloader configuration
        self.train_loader = DataLoader(self.init_attr('train', dataset), **self.config['loader'], shuffle=True)
        if self.config['valid'] is not None:
            self.valid_loader = DataLoader(self.init_attr('valid', dataset), **self.config['loader'], shuffle=False)
        # model configuration
        self.model = self.init_attr('model', network).cuda()
        # model parallel computing
        if self.config['trainer']['gpu_num'] > 1:
            self.model = DataParallel(self.model, device_ids=list(range(self.config['trainer']['gpu_num'])))
        # optimizer configuration
        self.optimizer = self.init_attr('optimizer', optim, self.model.parameters())
        # learning rate scheduler configuration
        self.lr_scheduler = self.init_attr('lr_scheduler', optim.lr_scheduler, self.optimizer)
        # logger configuration
        log_dir = os.path.join(self.config['trainer']['checkpoint_dir'], 'loggers', self.config['name'])
        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(log_dir, '{}-{}.txt'.format(self.config['trainer']['type'], str(self.date)))
        self.logger = Logger(log_filename)
        self.logger.write_dict(self.config)
        # loss functions configuration
        self.loss_func = getattr(loss, self.config['loss']['loss_func'])
        # resume model if specified
        if self.config['trainer']['resume']:
            self.resume_checkpoint()

    def init_attr(self, name, module, *mid_args):
        """
        retrieve the attribute based on name from module and use *args as arguments
        :param name: method or class name
        :param module: module or package that contains the name
        :param mid_args: arguments besides the kwargs
        :return: attr, retrieved attribute
        """
        attr = getattr(module, self.config[name]['type'])(*mid_args, **self.config[name]['args'])
        return attr

    def train(self):
        # clear all the content of the logger file
        self.logger.flush()
        # start training
        for epoch in range(1, self.config['trainer']['epoch_num'] + 1):
            self.logger.write_block(1)
            self.logger.write('Training EPOCH: {}'.format(str(epoch)))
            # train single one epoch
            avg_losses = self.run_epoch(epoch)
            self.epoch_log(avg_losses)
            # save model's checkpoint
            if epoch % self.config['trainer']['save_period'] == 0:
                self.save_checkpoint(epoch, avg_losses)
            # separate each saved epoch using two additional blocks
            self.logger.write_block(2)
            # start validation
            if self.valid_loader is not None:
                self.logger.write_block(1)
                self.logger.write('Validation EPOCH: {}'.format(str(epoch)))
                # valid single one epoch
                with torch.no_grad():
                    avg_losses = self.run_epoch(epoch, train=False)
                    self.epoch_log(avg_losses)

    def run_epoch(self, epoch_idx, train=True):
        epoch_losses = {}
        # set the model running mode
        if train:
            self.model.train()
            data_loader = self.train_loader
        else:
            self.model.eval()
            data_loader = self.valid_loader
        # start the training process
        for idx, batch in enumerate(tqdm(data_loader, desc=str(epoch_idx), unit='b')):
            images, gts = batch['image'].cuda(), batch['label'].cuda()
            output = self.model(images)
            # calculate losses
            losses = self.loss_func(images, output, self.config['loss']['config'])
            # step optimizer
            if train:
                if torch.isnan(losses['total_loss']):
                    raise ValueError('Loss Explosion!!!')
                else:
                    self.optimizer.zero_grad()
                    losses['total_loss'].backward()
                    self.optimizer.step()
                # learning rate decrease
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
            # accumulate losses through batches
            for key in losses.keys():
                if key not in epoch_losses.keys():
                    epoch_losses[key] = losses[key].detach().item() * images.size(0) / len(data_loader.dataset)
                else:
                    epoch_losses[key] += losses[key].detach().item() * images.size(0) / len(data_loader.dataset)
        return epoch_losses

    def save_checkpoint(self, epoch_idx, epoch_losses):
        """
        save trained model check point
        :param epoch_idx: the index of the epoch
        :param epoch_losses: losses log
        :return: None
        """
        state = {
            'epoch': epoch_idx,
            'configer': self.config,
            'model': (self.model.module if self.config['trainer']['gpu_num'] > 1 else self.model).state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': epoch_losses['total_loss']
        }
        chkpt_dir = os.path.join(self.config['trainer']['checkpoint_dir'], self.config['name'], str(self.date))
        os.makedirs(chkpt_dir, exist_ok=True)
        filename = os.path.join(chkpt_dir, '{}-{}-epoch-{}.pt'.format(self.config['trainer']['type'],
                                                                      epoch_idx, str(self.date)))
        self.logger.write("Saving checkpoint at: {} ...".format(filename))
        torch.save(state, filename)

    def resume_checkpoint(self):
        """
        resume the saved checkpoint
        :return: model checkpoint
        """
        self.logger.write("Loading checkpoint: {} ...".format(self.config['trainer']['resume_path']))
        checkpoint = torch.load(self.config['trainer']['resume_path'])
        # load model state dicts
        if checkpoint['configer']['model'] != self.config['model']:
            raise ValueError('Checkpoint Model Does Not Match to The Config File.')
        self.model.module.load_state_dict(checkpoint['model'])
        # load optimizer dicts
        if checkpoint['configer']['optimizer']['type'] != self.config['optimizer']['type']:
            raise ValueError('Checkpoint Optimizer Does Not Match to The Config File.')
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print('Optimizer resumed from before. Previous Loss: {}'.format(checkpoint['current_loss']))
        self.logger.write("Resume training from epoch {}".format(checkpoint['epoch']))

    def epoch_log(self, epoch_losses):
        # compute the average loss dicts and log
        for key in epoch_losses.keys():
            message = '{}: {}'.format(str(key), epoch_losses[key])
            self.logger.write(message)


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_file', type=str, default='./configs/drive/adaptive_lc.json')


if __name__ == '__main__':
    args = parser.parse_args()
    trainer = Trainer(args.config_file)
    trainer.train()
