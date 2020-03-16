import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from utils import *

"""
For continual learning. 
    |`- Transfer the features via convolutional layer
"""


class Sub_module(torch.nn.Module):
    def __init__(self, hiddens=[16 * 5 * 5, 200, 200], output_dim=10):
        super(Sub_module, self).__init__()

        self.dense_layers = torch.nn.ModuleList([])
        for idx in range(len(hiddens) - 1):
            self.dense_layers.append(torch.nn.Linear(hiddens[idx], hiddens[idx + 1]))

        self.output_layer = Density_estimator(hiddens[-1], output_dim, num_distr=10)

    def forward(self, x):
        x = x

        for idx in range(len(self.dense_layers)):
            x = self.dense_layers[idx](x).clamp(min=0)

        x = self.output_layer(x)

        return x




class Single_head(torch.nn.Module):
    def __init__(self, task_labels, transfer_learning=True):
        super(Single_head, self).__init__()
        self.conv_layers = Shared_conv()

        self.num_tasks = len(task_labels)
        self.columns = torch.nn.ModuleList([])
        for label in task_labels:
            self.columns.append(Sub_module(output_dim=len(label)))

    def forward(self, x, task_idx=0, return_probs=True, return_likelihood=True):
        z = self.conv_layers(x)

        return self.columns[task_idx](z)
    def parameters(self, col=None):
        if col is None:
            return super(Single_head, self).parameters()
        return self.columns[col].parameters()


    def freeze_conv(self,trainable=False):
        for param in self.conv_layers.parameters():
            param.requires_grad = trainable







class Multi_head(torch.nn.Module):
    def __init__(self, task_labels, transfer_learning=True):
        super(Multi_head, self).__init__()
        self.conv_layers = Shared_conv()
        self.estimator = Density_estimator(16 * 5 * 5, num_distr=10)

        self.num_tasks = len(task_labels)
        self.columns = torch.nn.ModuleList([])
        for label in task_labels:
            self.columns.append(Sub_module(output_dim=len(label)))

    def forward(self, x, task_idx=0, return_probs=True):
        z = self.conv_layers(x)
        classification = self.columns[task_idx](z)

        task_likelihood = self.estimator(z.detach(),return_probs)
        


        return classification, task_likelihood

    def parameters(self, col=None):
        if col is None:
            return super(Multi_head, self).parameters()
        return self.columns[col].parameters()


    def freeze_conv(self,trainable=False):
        for param in self.conv_layers.parameters():
            param.requires_grad = trainable










