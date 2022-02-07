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

from scipy import stats

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Reconstruction(BaseTrainer):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        # get classifier model and weights
        clf, weights, param_d, params_abs, signs, norms, lam_inv, state_d = self.get_nn_weights()
        self.nn_params = weights
        self.nn_state_d = state_d
        self.nn_clf = clf
        self.param_d = param_d
        self.params_abs = deepcopy(params_abs)
        self.params_res = deepcopy(params_abs)
        self.params_abs_recon = torch.zeros_like(params_abs)
        self.signs = signs
        self.norms = norms
        self.lam_inv = lam_inv
        self.n = len(params_abs)
        self.alpha = self.config.training.alpha
        self.scale_factor = np.log(float(self.n)/float(np.log(self.n)))
        self.alpha = self.alpha / self.scale_factor
        self.lam_inv = self.alpha * self.lam_inv

        self.gamma = self.config.training.gamma

        print("scale factor is: {}" .format(self.scale_factor))
        self.pruning = self.config.data.prune

        # saving
        self.checkpoint_dir = config.ckpt_dir
        self.output_dir = config.output_dir

    def get_model_cls(self, name):
        if name == 'cnn':
            model_cls = LeNet
        elif name == 'resnet18':
            model_cls = ResNet18
        elif name == 'vgg16':
            return VGG('VGG16')
        elif name == 'resnet20':
            return resnet20()
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
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9,
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
            norms = []
            print('Target network weights:')
            for (name, p) in model.named_parameters():
                if p.requires_grad:
                    weights = deepcopy(p.view(-1))
                    norms.append(torch.norm(weights) * torch.ones_like(weights))
                    if self.config.training.normalize:
                        weights = weights / torch.norm(weights)
                    # NOTE: let's squeeze everything together
                    params.append(weights)
                    print('{}: {}'.format(name, p.size()))
                    param_d[name] = p.size()
            # TODO: this may run out of memory if not careful
            params = torch.cat(params)
            norms = torch.cat(norms)

            #Save sign of the weights
            signs = torch.sign(params).float().cuda()
            params_abs = torch.abs(params)
            # compute mean of weight magnitudes
            lam_inv = torch.mean(params_abs)
            print('Mean of the magnitudes is: {}'.format(lam_inv))

        print('Total target network params: {}\n'.format(len(params)))

        return model, params, param_d, params_abs, signs, norms, lam_inv, checkpoint['state_dict']

    def load_reconst_weights(self, w_hat):
        i = 0
        w_hat = w_hat*self.signs
        if self.config.training.normalize:
            w_hat = w_hat*self.norms
        new_state_dict = deepcopy(self.nn_state_d)
        for k, k_shape in self.param_d.items():
            k_size = k_shape.numel()
            new_state_dict[k] = w_hat[i:(i + k_size)].view(k_shape)
            i += k_size
        self.nn_clf.load_state_dict(new_state_dict)

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

    def enc_step(self):
        param_list = self.params_res
        lambda_inv = self.lam_inv
        m_inds = torch.nonzero(param_list.gt(lambda_inv*self.scale_factor))

        if len(m_inds) == 0:
            print('no such index')
            return None, None
        else:
            m = np.random.choice(m_inds.detach().cpu().numpy().reshape(len(m_inds)))
            geom_rv = stats.geom(float(len(m_inds))/float(len(param_list)))  # Declare a geometric random variable
            k = geom_rv.rvs()  # Get a random sample from geom_rv, k to be used for bitrate estimates.

            # Update \hat{U}:
            self.params_abs_recon[m] = self.params_abs_recon[m] + self.lam_inv*self.scale_factor
            # Update U:
            self.params_res[m] = self.params_res[m] - self.lam_inv*self.scale_factor
            return m, k

    def train(self):
        best_loss = sys.maxsize
        best_acc = -sys.maxsize
        best_epoch = 1
        refresh_count = 0
        self.divide = 0
        self.loc = 0

        # save metrics
        recon_loss_db = np.zeros(self.config.training.n_iterations)
        indices = np.zeros(self.config.training.n_iterations)
        actual_indices = np.zeros(self.config.training.n_iterations)
        test_loss_db = np.zeros(int(self.config.training.n_iterations/self.config.data.period))
        test_clf_loss_db = np.zeros(int(self.config.training.n_iterations/self.config.data.period))
        test_clf_acc_db = np.zeros(int(self.config.training.n_iterations/self.config.data.period))
        pruning_db = np.zeros(int(self.config.training.n_iterations / self.config.data.period))

        for it in range(1, self.config.training.n_iterations + 1):
            m, k = self.enc_step()

            while m is None:
                refresh_count += 1
                if refresh_count % 20 == 0 and refresh_count>1:
                    self.alpha =  self.alpha*0.9
                # Refresh the parameter lambda.
                self.lam_inv = torch.mean(self.params_res)
                self.lam_inv = self.alpha * self.lam_inv
                # Compute m, k again after the parameter lambda is refreshed.
                m, k = self.enc_step()

            # The difference between the original and the reconstructed weights.
            recon_loss = torch.sum(torch.abs(self.params_abs-self.params_abs_recon))/self.n
            recon_loss_db[it - 1] = recon_loss
            if k is not None:
                indices[it - 1] = k     # for bitrate estimates.
                actual_indices[it - 1] = m      # the actual indices.

            if it % self.config.data.period == 0:
                print(f"iter = {it}, normalized_err = {recon_loss:.5f}, lam_inv = {self.lam_inv:.5f}")
                print('Number of refreshments: {}'.format(refresh_count))
                w_hat = deepcopy(self.params_abs_recon)
                # Load the reconstructed weights to the model.
                self.load_reconst_weights(w_hat)
                print('Finished loading reconstructed weights to target network')
                # Test the reconstructed model.
                test_clf_loss, test_clf_acc = self.test_clf()

                if best_acc <= test_clf_acc:
                    best_loss = test_clf_loss
                    best_acc = test_clf_acc
                    best_it = it

                # save metrics
                test_loss_db[int(it/self.config.data.period) - 1] = test_clf_loss
                test_clf_loss_db[int(it/self.config.data.period) - 1] = test_clf_loss
                test_clf_acc_db[int(it/self.config.data.period) - 1] = test_clf_acc
                pruning_db[int(it / self.config.data.period) - 1] = 1- len(torch.nonzero(self.params_abs_recon)) / len(self.params_abs_recon)
                print('%.2f sparse' % (1 - len(torch.nonzero(self.params_abs_recon)) / len(self.params_abs_recon)))
                np.save(os.path.join(self.output_dir, 'tr_loss.npy'), recon_loss_db)
                np.save(os.path.join(self.output_dir, 'indices.npy'), indices)
                np.save(os.path.join(self.output_dir, 'actual_indices.npy'), actual_indices)
                np.save(os.path.join(self.output_dir, 'test_loss.npy'), test_loss_db)
                np.save(os.path.join(self.output_dir, 'test_clf_loss.npy'), test_clf_loss_db)
                np.save(os.path.join(self.output_dir, 'test_clf_acc.npy'), test_clf_acc_db)
                np.save(os.path.join(self.output_dir, 'pruning.npy'), pruning_db)

                # Check if we reach the target sparsity.
                if pruning_db[int(it / self.config.data.period) - 1] <= self.pruning[-1]:
                    if not os.path.exists(self.config.training.mask_dir):
                        os.makedirs(self.config.training.mask_dir)
                    torch.save(self.params_abs_recon, os.path.join(self.config.training.mask_dir,
                                                               'pruned_{}_abs.pt'.format(self.pruning[-1])),
                           _use_new_zipfile_serialization=False)
                    torch.save(self.signs, os.path.join(self.config.training.mask_dir,
                                                    'pruned_{}_signs.pt'.format(self.pruning[-1])),
                           _use_new_zipfile_serialization=False)
                    if self.config.training.normalize:
                        torch.save(self.norms, os.path.join(self.config.training.mask_dir,
                                                    'pruned_{}_norms.pt'.format(self.pruning[-1])),
                                                    _use_new_zipfile_serialization=False)
                    self.pruning = self.pruning[:-1]
                    if self.pruning==[]:
                        break

            # Update lambda at every iteration.
            self.gamma = (self.n - 1)/(self.n - self.scale_factor)
            self.lam_inv = self.gamma*(self.n - self.scale_factor) / self.n * self.lam_inv

        w_orig = deepcopy(self.params_abs)
        self.load_reconst_weights(w_orig)
        print('Finished loading original weights to target network')
        test_loss, test_acc = self.test_clf()
        print(test_acc)
        print('Completed training! Best performance at iteration {}, loss: {}, acc: {}'.format(
            best_it,
            np.round(best_loss, 5),
            np.round(best_acc, 5)))


    def test(self):
        _, dataloader = self.get_clf_dataset()

        # get meters ready
        loss_meter = utils.AverageMeter()
        acc_meter = utils.AverageMeter()
        summary = {'avg_loss': 0, 'avg_acc': 0}

        self.nn_state_d.eval()

        with torch.no_grad():
            # test classifier
            t = tqdm(iter(dataloader), leave=False, total=len(dataloader))
            for i, (x, y) in enumerate(t):
                x = x.to(device).float()
                x = x.view(-1, self.config.data.channels, self.config.data.image_size, self.config.data.image_size)
                y = y.to(device).long()

                # classification loss
                y_logits = self.nn_state_d(x)
                loss = F.cross_entropy(y_logits, y)
                loss_meter.update(loss)

                # check accuracy
                accs = self.accuracy(y_logits, y)
                acc_meter.update(accs)

        # Completed running test
        print('Completed evaluation: test loss: {}, test acc: {}'.format(
            np.round(loss_meter.avg.item(), 5),
            np.round(acc_meter.avg.item(), 5)))
        summary.update(
            dict(avg_loss=np.round(loss_meter.avg.item(), 5),
                 avg_acc=np.round(acc_meter.avg.item(), 5)))
        print()
        # pprint(summary)

        return loss_meter.avg.item(), acc_meter.avg.item()

    def test_clf(self):
        _, dataloader = self.get_clf_dataset()

        # get meters ready
        loss_meter = utils.AverageMeter()
        acc_meter = utils.AverageMeter()
        summary = {'avg_loss': 0, 'avg_acc': 0}

        self.nn_clf.eval()

        with torch.no_grad():
            # test classifier
            t = tqdm(iter(dataloader), leave=False, total=len(dataloader))
            for i, (x, y) in enumerate(t):
                x = x.to(device).float()
                x = x.view(-1, self.config.data.channels, self.config.data.image_size, self.config.data.image_size)
                y = y.to(device).long()

                # classification loss
                y_logits = self.nn_clf(x)
                loss = F.cross_entropy(y_logits, y)
                loss_meter.update(loss)

                # check accuracy
                accs = self.accuracy(y_logits, y)
                acc_meter.update(accs)

        # Completed running test
        print('Completed evaluation: test loss: {}, test acc: {}'.format(
            np.round(loss_meter.avg.item(), 5),
            np.round(acc_meter.avg.item(), 5)))
        summary.update(
            dict(avg_loss=np.round(loss_meter.avg.item(), 5),
                 avg_acc=np.round(acc_meter.avg.item(), 5)))
        print()
        # pprint(summary)

        return loss_meter.avg.item(), acc_meter.avg.item()