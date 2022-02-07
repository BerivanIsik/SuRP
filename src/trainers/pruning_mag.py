import os
import sys

from copy import deepcopy
import logging
from pprint import pprint
from tqdm import tqdm
import numpy as np

import datasets as dsets
import torch.utils.data as data_utils
import utils
from models import *
from trainers.base import BaseTrainer

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms
import torch.utils.data as data_utils

import matplotlib

matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Prune(BaseTrainer):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        # get classifier model and weights
        clf, weights, param_d, state_d = self.get_nn_weights()
        self.nn_params = weights
        self.nn_state_d = state_d
        self.nn_clf = clf
        self.param_d = param_d

        # model and optimizer
        #self.model = self.get_model()
        self.optimizer = self.get_optimizer(self.nn_clf.parameters())
        self.loss = self.get_loss()

        # saving
        self.checkpoint_dir = config.ckpt_dir
        self.output_dir = config.output_dir

        #data
        self.train_dataloader, self.test_dataloader = self.get_clf_dataset()

    def get_model_cls(self, name):
        if name == 'cnn':
            model_cls = LeNet
        elif name == 'resnet18':
            model_cls = ResNet18
        elif name == 'vgg16':
            return VGG('VGG16')
        else:
            print('Model {} not found!'.format(name))
            raise NotImplementedError

        return model_cls(self.config)

    def get_optimizer(self, parameters):
        if self.config.optim.optimizer == 'Adam':
            return optim.Adam(
                parameters, lr=self.config.optim.lr_retrain,
                weight_decay=self.config.optim.weight_decay,
                betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad)
        elif self.config.optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.config.optim.lr_retrain, momentum=0.9,
                             weight_decay=self.config.optim.weight_decay)
        else:
            raise NotImplementedError()

    def get_clf_dataset(self):
        # get data transformations
        train, test = dsets.get_dataset(self.config)

        # create dataloaders
        train = data_utils.DataLoader(train, batch_size=self.config.training.batch_size, shuffle=True)
        test = data_utils.DataLoader(test, batch_size=self.config.training.batch_size, shuffle=False)

        return train, test

    def get_nn_weights(self):
        """Get weights for original MLP trained on MNIST
        """
        param_d = {}
        with torch.no_grad():
            model = self.get_model_cls(self.config.nn.name)
            model = model.to(self.config.device)

            # resume best checkpoint weights
            resume_path = os.path.join(self.config.clf_ckpt_dir, 'model_best.pth')

            checkpoint = torch.load(resume_path)
            model.load_state_dict(checkpoint['state_dict'])

            # grab weights
            model.eval()
            params = []
            print('Target network weights:')
            for (name, p) in model.named_parameters():
                if p.requires_grad:
                    # NOTE: let's squeeze everything together
                    params.append(p.view(-1))
                    print('{}: {}'.format(name, p.size()))
                    param_d[name] = p.size()
            # TODO: this may run out of memory if not careful
            params = torch.cat(params)

        print('Total target network params: {}\n'.format(len(params)))

        return model, params, param_d, checkpoint['state_dict']


    def get_model(self):
        model = self.get_model_cls(self.config.model.name)
        model = model.to(self.config.device)

        return model

    def accuracy(self, logits, y):
        with torch.no_grad():
            probas = F.softmax(logits, dim=1)
            _, y_preds = torch.max(probas, 1)
            acc = (y_preds == y).sum()
            acc = torch.true_divide(acc, len(y_preds)).cpu().numpy()

        return acc


    def retrain_epoch(self, epoch):
        # get meters ready
        loss_meter = utils.AverageMeter()
        acc_meter = utils.AverageMeter()

        # train classifier
        self.nn_clf.train()
        t = tqdm(iter(self.train_dataloader), leave=False, total=len(self.train_dataloader))
        for i, (x, y) in enumerate(t):
            x = x.to(device).float()
            x = x.view(-1, self.config.data.channels, self.config.data.image_size, self.config.data.image_size)
            y = y.to(device).long()

            # classification loss
            self.optimizer.zero_grad()
            y_logits = self.nn_clf(x)
            loss = F.cross_entropy(y_logits, y)
            loss_meter.update(loss)

            # gradient update
            loss.backward()

            # check accuracy
            accs = self.accuracy(y_logits, y)
            acc_meter.update(accs)

            for k, m_net in enumerate(self.nn_clf.modules()):
                if isinstance(m_net, nn.Conv2d) or isinstance(m_net, nn.Linear):
                    weights = m_net.weight.data.abs().clone()
                    mask = weights.gt(0).float().cuda()
                    m_net.weight.grad.data.mul_(mask)

            # update params
            self.optimizer.step()

            # get summary
            summary = dict(avg_loss=loss.item(), clf_acc=accs.item())

            if (i + 1) % self.config.training.iter_log == 0:
                summary.update(
                    dict(avg_loss=np.round(loss.float().item(), 3),
                         clf_acc=np.round(accs, 3)))

        # end of training epoch
        print('Completed epoch {}: train loss: {}, train acc: {}'.format(
            epoch,
            np.round(loss_meter.avg.item(), 3),
            np.round(acc_meter.avg.item(), 3)))
        summary.update(dict(
            avg_loss=loss_meter.avg.item(),
            avg_acc=acc_meter.avg.item()))

        return loss_meter.avg.item(), acc_meter.avg.item()

    def retrain(self):
        print('Test classification performance of pruned network:')
        self.test()

        # adjust learning rate as you go
        #scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[150, 250], gamma=0.1)

        best_acc = -sys.maxsize
        best_loss = sys.maxsize
        for epoch in range(1, self.config.training.retrain_epochs + 1):
            print('retraining epoch {}'.format(epoch))
            tr_loss, tr_acc = self.retrain_epoch(epoch)
            test_loss, test_acc = self.test()

            if test_acc >= best_acc:
                best_acc = test_acc
                best_epoch = epoch
                best = True
                if test_loss <= best_loss:
                    best_loss = test_loss
            else:
                best = False
        print('Completed training! Best performance at epoch {}, loss: {}, acc: {}'.format(best_epoch, best_loss,
                                                                                           best_acc))

    def pruning(self, ratio):
        total = 0
        for m in self.nn_clf.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                total += m.weight.data.numel()
        conv_weights = torch.zeros(total)
        conv_weights_norm = torch.zeros(total)
        norms = []
        index = 0
        for m in self.nn_clf.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                size = m.weight.data.numel()
                conv_weights[index:(index + size)] = m.weight.data.view(-1).abs().clone()
                if self.config.training.normalize:
                    weights = deepcopy(m.weight.data.view(-1)).abs().clone()
                    norm = torch.norm(weights)
                    norms.append(norm)
                    conv_weights_norm[index:(index + size)] = weights.clone()/norm

                index += size
        if self.config.training.normalize:
            y, i = torch.sort(conv_weights_norm)
        else:
            y, i = torch.sort(conv_weights)

        thresh_index = int(total * ratio)
        threshold = y[thresh_index]
        threshold = threshold.cuda()
        pruned = 0
        print('Pruning threshold is: {}% at {}'.format(ratio * 100, threshold))
        layer = 0
        for m in self.nn_clf.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                weight_copy = m.weight.data.abs().clone()
                if self.config.training.normalize:
                    mask = weight_copy.gt(threshold*norms[layer]).float().cuda()
                else:
                    mask = weight_copy.gt(threshold).float().cuda()
                pruned = pruned + mask.numel() - torch.sum(mask)
                # mask out original weights
                m.weight.data.mul_(mask)
                layer+=1
        print('Total conv params: {}, Prunned conv params: {}, Pruned ratio: {}'.format(total, pruned, pruned / total))


    def train(self):
        pruning = np.array(self.config.data.prune)
        test_clf_loss_db = np.zeros(len(pruning))
        test_clf_acc_db = np.zeros(len(pruning))
        test_clf_acc_db_retrained = np.zeros(len(pruning))
        self.test()
        count = 0
        # Go over all the sparsity values in pruning.
        for ratio in pruning:
            clf, weights, param_d, state_d = self.get_nn_weights()
            self.nn_clf = clf
            self.optimizer = self.get_optimizer(self.nn_clf.parameters())
            self.pruning(ratio)
            test_loss, test_acc = self.test()
            test_clf_loss_db[count] = test_loss
            test_clf_acc_db[count] = test_acc
            self.retrain()
            test_loss_retrained, test_acc_retrained = self.test()
            test_clf_acc_db_retrained[count] = test_acc_retrained
            count += 1
        np.save(os.path.join(self.output_dir, 'pruned_acc.npy'), test_clf_acc_db)
        np.save(os.path.join(self.output_dir, 'retrained_pruned_acc.npy'), test_clf_acc_db_retrained)
        np.save(os.path.join(self.output_dir, 'prune_ratios.npy'), pruning)

    def test(self):
        _, dataloader = self.get_clf_dataset()
        # get meters ready
        loss_meter = utils.AverageMeter()
        acc_meter = utils.AverageMeter()
        summary = {'avg_loss': 0, 'avg_acc': 0}
        test_type = 'test'

        self.model = self.nn_clf
        model = self.model

        with torch.no_grad():
            # test classifier
            model.eval()
            # test classifier
            t = tqdm(iter(dataloader), leave=False, total=len(dataloader))
            for i, (x, y) in enumerate(t):
                x = x.to(device).float()
                x = x.view(-1, self.config.data.channels, self.config.data.image_size, self.config.data.image_size)
                y = y.to(device).long()

                # classification loss
                y_logits = model(x)
                loss = F.cross_entropy(y_logits, y)
                loss_meter.update(loss)

                # check accuracy
                accs = self.accuracy(y_logits, y)
                acc_meter.update(accs)

        # Completed running test
        params = list(model.parameters())
        params = torch.cat([p.view(-1) for p in params])
        print('%.2f sparse' % (1 - len(torch.nonzero(params)) / len(params)))
        print('Completed evaluation: {} loss: {}, {} acc: {}'.format(
            test_type,
            np.round(loss_meter.avg.item(), 3),
            test_type,
            np.round(acc_meter.avg.item(), 3)))
        summary.update(
            dict(avg_loss=np.round(loss_meter.avg.item(), 3),
                 avg_acc=np.round(acc_meter.avg.item(), 3)))

        return loss_meter.avg.item(), acc_meter.avg.item()