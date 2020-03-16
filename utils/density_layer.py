import torch.nn.functional as F
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn

    

class Shared_conv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        return x


class PNN(torch.nn.Module):
    def __init__(self, in_features, out_features, num_distr=1):
        super().__init__()
        self.centers = {}
        self.out_features = out_features
        self.num_distr = num_distr
        for i in range(out_features):
            self.centers[i] = []
            for idx in range(num_distr):
                params = torch.nn.Parameter(torch.zeros(in_features, requires_grad=True))
                self.register_parameter('center%d%d' % (i,idx), params)
                self.centers[i].append(params)

    def forward(self, x):
        outputs = []
        for out_idx in range(self.out_features):
            probs = []
            for distr_idx in range(self.num_distr):
                probs.append(self.gaussian_activation(((x - self.centers[out_idx][distr_idx]) ** 2).sum(dim=-1)))

            probs = torch.transpose(torch.stack(probs), 0, 1)
            probs = torch.sum(probs, dim=-1, keepdim=False) / (torch.sum(probs, dim=-1, keepdim=False) \
                                                                  + self.num_distr - self.num_distr * torch.max(probs, dim=-1, keepdim=False)[0])
            outputs.append(probs)

        outputs = torch.transpose(torch.stack(outputs), 0, 1)
        return outputs

    def gaussian_activation(self, x, sigma=0.5):
        return torch.exp(-x / (2 * sigma * sigma))



class Density_estimator(torch.nn.Module):
    def __init__(self, in_features, out_features=200, num_distr=1):
        super().__init__()
        self.centers = []
        self.num_distr = num_distr
        self.dense = nn.Linear(in_features,out_features)
        for i in range(num_distr):
            mean = torch.nn.Parameter(torch.rand(out_features, requires_grad=True))
            self.register_parameter('mean%d' % i, mean)
            rho = torch.nn.Parameter(torch.rand(out_features, requires_grad=True))
            sigma = torch.log(1 + torch.exp(rho))
            self.register_parameter('rho%d' % i, rho)

            self.centers.append([mean,sigma])

    def forward(self, x, return_probs=False):
        probs = []
        x = self.dense(x)
        for c in self.centers:
            estimate = (x - c[0])**2 / c[1]
            likelihood = self.gaussian_activation(estimate)
            probs.append(likelihood)

        probs = torch.transpose(torch.stack(probs), 0, 1)

        if return_probs:
            probs = torch.sum(probs, dim=-1, keepdim=True) / (torch.sum(probs, dim=-1, keepdim=True) \
                                                                  + self.num_distr - self.num_distr * torch.max(probs, dim=-1, keepdim=True)[0])

        return probs

    def gaussian_activation(self, x):
        return torch.exp(-torch.sum(x,dim=-1))
















