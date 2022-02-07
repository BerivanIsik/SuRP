import os
import sys

import copy
import logging
from pprint import pprint
from tqdm import tqdm
import numpy as np
from decimal import *
import datasets as dsets
import torch.utils.data as data_utils
import utils
from models import *
from trainers.base import BaseTrainer

import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms
import torch.utils.data as data_utils

import matplotlib
matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Classifier(BaseTrainer):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        # model and optimizer
        self.model = self.get_model()
        self.gradient = self.get_model()
        self.optimizer = self.get_optimizer(self.model.parameters())
        print('Number of model parameters: {}'.format(
            utils.count_params(self.model)))

        # saving
        self.checkpoint_dir = os.path.join(config.training.save_dir, config.training.exp_id)

    def get_model_cls(self, name):
        if name == 'mlp':
            model_cls = MLP
        elif name == 'resnet18':
            model_cls = ResNet18
        elif name == 'resnet34':
            model_cls = ResNet34
        elif name == 'vgg16':
            return VGG('VGG16')
        elif name == 'resnet20':
            return resnet20()
        elif name == 'cnn':
            model_cls = LeNet
        elif name == 'LeNet_300_100':
            return LeNet_300_100()
        else:
            print('Model {} not found!'.format(name))
            raise NotImplementedError

        return model_cls(self.config)

    def get_optimizer(self, parameters):
        if self.config.optim.optimizer == 'Adam':
            return optim.Adam(
                parameters, lr=self.config.optim.lr, 
                weight_decay=self.config.optim.weight_decay, 
                betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad)
        elif self.config.optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9, weight_decay=self.config.optim.weight_decay)
        else:
            raise NotImplementedError()

    def get_dataset(self):
        # get data transformations
        train, test = dsets.get_dataset(self.config)

        # create dataloaders
        train = data_utils.DataLoader(train, batch_size=self.config.training.batch_size, shuffle=True)
        test = data_utils.DataLoader(test, batch_size=self.config.training.batch_size, shuffle=False)

        return train, test

    def get_model(self):
        model = self.get_model_cls(self.config.model.name)
        model = model.to(self.config.device)

        return model

    def accuracy(self, logits, y):
        with torch.no_grad():
            _, y_preds = torch.max(logits, 1)
            acc = (y_preds == y).sum()
            acc = torch.true_divide(acc, len(y_preds)).cpu().numpy()
        
        return acc

    def train_epoch(self, epoch):
        train_dataloader, test_dataloader = self.get_dataset()

        # get meters ready
        loss_meter = utils.AverageMeter()
        acc_meter = utils.AverageMeter()

        # train classifier
        self.model.train()
        t = tqdm(iter(train_dataloader), leave=False, total=len(train_dataloader))
        for i, (x, y) in enumerate(t):
            x = x.to(device).float()
            #x = x.view(-1, self.config.data.in_dim)
            x = x.view(-1, self.config.data.channels, self.config.data.image_size, self.config.data.image_size)
            #x = x.view(-1, 1, 28, 28)
            y = y.to(device).long()

            # classification loss
            y_logits = self.model(x)
            loss = F.cross_entropy(y_logits, y)
            loss_meter.update(loss)

            # check accuracy
            accs = self.accuracy(y_logits, y)
            acc_meter.update(accs)

            # gradient update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # get summary
            summary = dict(avg_loss=loss.item(), clf_acc=accs.item())

            if (i + 1) % self.config.training.iter_log == 0:
                summary.update(
                    dict(avg_loss=np.round(loss.float().item(), 3),
                        clf_acc=np.round(accs, 3)))
                print()
                pprint(summary)

            # pbar
            desc = f'loss {loss:.4f}'
            t.set_description(desc)
            t.update(x.shape[0])
            # if (i + 1) % self.config.training.iter_save == 0:
            #     print("Saved to", self.config.ckpt_dir)
            #     self._save_checkpoint(epoch)
        # end of training epoch
        print()
        print('Completed epoch {}: train loss: {}, train acc: {}'.format(
            epoch, 
            np.round(loss_meter.avg.item(), 3), 
            np.round(acc_meter.avg.item(), 3)))
        summary.update(dict(
            avg_loss=loss_meter.avg.item(),
            avg_acc=acc_meter.avg.item()))
        # pprint(summary)

        return loss_meter.avg.item(), acc_meter.avg.item()

    def train(self):
        best = False
        best_loss = sys.maxsize
        best_acc = -sys.maxsize
        best_epoch = 1
        tr_loss_db = np.zeros(self.config.training.n_epochs)
        test_loss_db = np.zeros(self.config.training.n_epochs)

        if self.config.model.name == 'LeNet_300_100':
            scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[25, 50,75], gamma=0.2)
        elif self.config.model.name == 'resnet18':
            scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[150, 250], gamma=0.1)
        elif self.config.model.name == 'vgg16':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        elif self.config.model.name == 'resnet20':
            scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[150, 250], gamma=0.1)

        params = list(self.model.parameters())
        params = torch.cat([p.view(-1) for p in params])
        print('Total number of parameters: {}'  .format(len(params)))
        for epoch in range(1, self.config.training.n_epochs+1):
            print('training epoch {}'.format(epoch))
            tr_loss, tr_acc = self.train_epoch(epoch)
            test_loss, test_acc = self.test()
            if self.config.data.dataset == 'cifar10' or self.config.model.name == 'LeNet_300_100':
                scheduler.step()
            if test_loss <= best_loss:
                best_loss = test_loss
                best_acc = test_acc
                best_epoch = epoch
                self._save_checkpoint(epoch, save_best=True)
            # save metrics
            tr_loss_db[epoch-1] = tr_loss
            test_loss_db[epoch-1] = test_loss

            # checkpoint model every <iter_save> epochs
            if (epoch) % self.config.training.iter_save == 0:
                print("Saved to", self.config.ckpt_dir)
                self._save_checkpoint(epoch)
        print('Completed training! Best performance at epoch {}, loss: {}, acc: {}'.format(best_epoch, best_loss, best_acc))

    def test(self):
        _, dataloader = self.get_dataset()

        # get meters ready
        loss_meter = utils.AverageMeter()
        acc_meter = utils.AverageMeter()
        summary = {'avg_loss': 0, 'avg_acc': 0}

        with torch.no_grad():
            # test classifier
            self.model.eval()
            t = tqdm(iter(dataloader), leave=False, total=len(dataloader))
            for i, (x, y) in enumerate(t):
                x = x.to(device).float()
                # x = x.view(-1, self.config.data.in_dim)
                x = x.view(-1, self.config.data.channels, self.config.data.image_size, self.config.data.image_size)
                y = y.to(device).long()

                # classification loss
                y_logits = self.model(x)
                loss = F.cross_entropy(y_logits, y)
                loss_meter.update(loss)

                # check accuracy
                accs = self.accuracy(y_logits, y)
                acc_meter.update(accs)
        # Completed running test
        print('Completed evaluation: test loss: {}, test acc: {}'.format(
            np.round(loss_meter.avg.item(), 3), 
            np.round(acc_meter.avg.item(), 3)))
        summary.update(
            dict(avg_loss=np.round(loss_meter.avg.item(), 3),
                avg_acc=np.round(acc_meter.avg.item(), 3)))
        print()
        # pprint(summary)

        return loss_meter.avg.item(), acc_meter.avg.item()
