import os
import numpy as np

import torch
import torchvision.transforms as transforms
from torchvision.datasets import (
    MNIST,
    CIFAR10
)
from datasets.dataset_utils import *


def get_dataset(config, params=None):
    if config.data.dataset not in ['synthetic', 'logreg', 'lr_weights']:
        train_transform, test_transform = get_data_transform(config)

    if config.data.dataset == 'synthetic':
        if params is None:
            raise NotImplementedError
        dataset = WeightSequence(data=params, split='train')
        test_dataset = WeightSequence(data=params, split='test')

    elif config.data.dataset == 'logreg':
        dataset = Synthetic(config=config, split='train')
        test_dataset = Synthetic(config=config, split='test')

    elif config.data.dataset == 'lr_weights':
        weights = LRWeights(config=config)
        # i am squishing all of the data together
        dataset = SyntheticConcat(config=config, split='train')
        test_dataset = SyntheticConcat(config=config, split='test')

        # TODO: returning 3 things instead of 2!
        return weights, dataset, test_dataset

    elif config.data.dataset == 'mlp_weights':
        dataset = MNIST(
            root=config.training.data_dir, 
            train=True, download=True, transform=train_transform)
        test_dataset = MNIST(
            root=config.training.data_dir, train=False, download=True, transform=test_transform)

    elif config.data.dataset == 'old_mlp_weights':
        weights = MLPWeights(config=config)
        dataset = MNIST(
            root=config.training.data_dir, 
            train=True, download=True, transform=train_transform)
        test_dataset = MNIST(
            root=config.training.data_dir, train=False, download=True, transform=test_transform)

        # TODO: returning 3 things instead of 2!
        return weights, dataset, test_dataset

    elif config.data.dataset == 'mnist':
        dataset = MNIST(
            root=config.training.data_dir, 
            train=True, download=True, transform=train_transform)
        test_dataset = MNIST(
            root=config.training.data_dir, train=False, download=True, transform=test_transform)

    elif config.data.dataset == 'omniglot':
        raise NotImplementedError

    elif config.data.dataset == 'cifar10':
        dataset = CIFAR10(os.path.join(config.training.data_dir, 'cifar10'), train=True, download=True,
                          transform=train_transform)
        test_dataset = CIFAR10(os.path.join(config.training.data_dir, 'cifar10_test'), train=False, download=True,
                               transform=test_transform)

    elif config.data.dataset == 'celeba':
          dataset = CelebA(config=config, split='train', transform=train_transform)
          test_dataset = CelebA(config=config, split='val', transform=test_transform)

    else:
        raise NotImplementedError

    return dataset, test_dataset


def get_data_transform(config):
    if config.data.dataset == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                (0.2023, 0.1994, 0.2010)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                (0.2023, 0.1994, 0.2010)),
        ])

    else:
        # everything else atm?
        train_transform = test_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.ToTensor()
        ])
    return train_transform, test_transform