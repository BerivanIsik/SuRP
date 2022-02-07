import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
  def __init__(self, config):
      super(LeNet, self).__init__()
      self.config = config

      self.conv1 = nn.Conv2d(1, 20, 5)
      self.conv2 = nn.Conv2d(20, 50, 5)
      
      # an affine operation: y = Wx + b
      self.fc1 = nn.Linear(50 * 4 * 4, 500)  # 4*4 from image dimension
      self.fc2 = nn.Linear(500, 10)

  def forward(self, x):
      # Max pooling over a (2, 2) window
      x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # (100, 20, 12, 12)
      # If the size is a square you can only specify a single number
      x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # (100, 50, 4, 4)
      x = x.view(-1, self.num_flat_features(x))
      x = F.relu(self.fc1(x))
      x = self.fc2(x)

      return x  # logits

  def num_flat_features(self, x):
      size = x.size()[1:]  # all dimensions except the batch dimension
      num_features = 1
      for s in size:
          num_features *= s
      return num_features


class LeNet_300_100(nn.Module):
    def __init__(self):
        super(LeNet_300_100, self).__init__()
        self.ip1 = nn.Linear(28*28, 300)
        self.relu_ip1 = nn.ReLU(inplace=True)
        self.ip2 = nn.Linear(300, 100)
        self.relu_ip2 = nn.ReLU(inplace=True)
        self.ip3 = nn.Linear(100, 10)
        return

    def forward(self, x):
        x = x.view(x.size(0), 28*28)
        x = self.ip1(x)
        x = self.relu_ip1(x)
        x = self.ip2(x)
        x = self.relu_ip2(x)
        x = self.ip3(x)
        return x