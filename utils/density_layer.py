import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn


class Shared_conv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        return x


class PNN(torch.nn.Module):
    def __init__(self, in_features, out_features, num_distr=1):
        super().__init__()
        self.training = True
        self.centers = {}
        self.out_features = out_features
        self.num_distr = num_distr
        for i in range(out_features):
            self.centers[i] = []
            for idx in range(num_distr):
                params = torch.nn.Parameter(torch.rand(in_features, requires_grad=True))
                self.register_parameter('center%d%d' % (i, idx), params)
                self.centers[i].append(params)

    def forward(self, x):
        outputs = []
        for out_idx in range(self.out_features):
            probs = []
            for distr_idx in range(self.num_distr):
                probs.append(self.gaussian_activation(((x - self.centers[out_idx][distr_idx]) ** 2).sum(dim=-1)))

            probs = torch.stack(probs, 1)
            ##################
            if self.training:
                # diff = self.num_distr * torch.max(probs,dim=-1)[0] - torch.sum(probs,dim=-1)
                # probs = diff / (diff + self.num_distr - self.num_distr * torch.max(probs, dim=-1, keepdim=False)[0])
                probs = (torch.max(probs, dim=-1)[0] * (self.num_distr + 1) - torch.sum(probs, dim=-1)) / self.num_distr
                ##################
            else:
                probs = torch.sum(probs, dim=-1) / (torch.sum(probs, dim=-1)
                                                    + self.num_distr - self.num_distr * torch.max(probs, dim=-1)[0])
            outputs.append(probs)

        outputs = torch.stack(outputs, 1)
        return outputs

    def gaussian_activation(self, x, sigma=1.):
        return torch.exp(-x / (2 * sigma * sigma))


class Density_estimator(torch.nn.Module):
    def __init__(self, in_features, out_features=200, num_distr=1):
        super().__init__()
        self.training = False
        self.centers = {}
        self.num_distr = num_distr
        self.in_features = in_features
        self.out_features = out_features
        # self.dense = nn.Linear(in_features,out_features)
        for i in range(out_features):
            self.centers[i] = []
            for idx in range(num_distr):
                mean = torch.nn.Parameter(torch.rand(in_features, requires_grad=True))
                self.register_parameter('mean%d%d' % (idx, i), mean)
                rho = torch.nn.Parameter(torch.rand(in_features, requires_grad=True))
                self.register_parameter('rho%d%d' % (idx, i), rho)
                self.centers[i].append([mean, rho])

    def forward(self, x):
        outputs = []
        # self.reg = []
        for out_idx in range(self.out_features):
            probs = []
            for distr_idx in range(self.num_distr):
                sigma = torch.log(1 + torch.exp(self.centers[out_idx][distr_idx][1]))
                estimate = (x - self.centers[out_idx][distr_idx][0]) ** 2 / (2 * sigma * sigma)
                probs.append(self.gaussian_activation(estimate))

            probs = torch.stack(probs, 1)

            ##################
            if self.training:
                # diff = self.num_distr * torch.max(probs,dim=-1)[0] - torch.sum(probs,dim=-1)
                # probs = diff / (diff + self.num_distr - self.num_distr * torch.max(probs, dim=-1, keepdim=False)[0])
                probs = (torch.max(probs, dim=-1)[0] * (self.num_distr + 1) - torch.sum(probs, dim=-1)) / self.num_distr
                # self.reg.append(1-(torch.max(probs,dim=-1)[0] * (self.num_distr+1) - torch.sum(probs,dim=-1))/self.num_distr )
            ##################
            else:
                probs = torch.sum(probs, dim=-1) / (torch.sum(probs, dim=-1)
                                                    + self.num_distr - self.num_distr * torch.max(probs, dim=-1)[0])
            outputs.append(probs)
        # self.reg = torch.stack(self.reg,1)
        outputs = torch.stack(outputs, 1)
        return outputs

    def get_distr_index(self, x):
        outputs = []
        for out_idx in range(self.out_features):
            probs = []
            for distr_idx in range(self.num_distr):
                sigma = torch.log(1 + torch.exp(self.centers[out_idx][distr_idx][1]))
                estimate = (x - self.centers[out_idx][distr_idx][0]) ** 2 / (2 * sigma * sigma)
                probs.append(self.gaussian_activation(estimate))
            probs = torch.stack(probs, 1)
            outputs.append(torch.argmax(probs, dim=-1))
        outputs = torch.stack(outputs, 1)
        return outputs

    def gaussian_activation(self, x):
        return torch.exp(-torch.sum(x, dim=-1))


class Dynamic_estimator(torch.nn.Module):
    def __init__(self, in_features, out_features=200, num_distr=1):
        super().__init__()
        self.training = True
        self.centers = {}
        self.num_distr = num_distr
        self.in_features = in_features
        self.out_features = out_features
        # self.dense = nn.Linear(in_features,out_features)
        self.training_centers = {}
        for i in range(out_features):
            self.centers[i] = []
            self.training_centers[i] = []
            for idx in range(num_distr):
                mean = torch.nn.Parameter(torch.rand(in_features, requires_grad=True))
                self.register_parameter('mean%d%d' % (idx, i), mean)
                rho = torch.nn.Parameter(torch.rand(in_features, requires_grad=True))
                self.register_parameter('rho%d%d' % (idx, i), rho)
                self.centers[i].append([mean, rho])
                if idx == 0:
                    self.training_centers[i].append(True)
                else:
                    self.training_centers[i].append(False)

    def set_training_center(self, class_idx, center_idx):
        for idx, c in enumerate(self.training_centers[class_idx]):
            if idx == center_idx:
                self.training_centers[class_idx][idx] = True
            else:
                self.training_centers[class_idx][idx] = False

    def forward(self, x):
        outputs = []
        # self.reg = []
        for out_idx in range(self.out_features):
            probs = []
            training_idx = np.argmax(self.training_centers[out_idx])
            for distr_idx in range(self.num_distr):
                sigma = torch.log(1 + torch.exp(self.centers[out_idx][distr_idx][1]))
                estimate = (x - self.centers[out_idx][distr_idx][0]) ** 2 / (2 * sigma * sigma)
                P = self.gaussian_activation(estimate)
                if self.training:
                    if distr_idx == training_idx:
                        probs.append(P)
                else:
                    probs.append(P)

            if self.training:
                outputs.append(probs[0])
            else:
                probs = torch.stack(probs, 1)
                probs = torch.max(probs,1)[0]
                # probs = torch.sum(probs, dim=-1) / (torch.sum(probs, dim=-1)
                #                                   + self.num_distr - self.num_distr * torch.max(probs, dim=-1)[0])
                outputs.append(probs)
        outputs = torch.stack(outputs, 1)
        return outputs

    # uti
    # agitaion
    # activaties
    # MCM to create the entropy
    # keep updating based on the activaties of the users
    # Find the change point

    def get_distr_index(self, x):
        outputs = []
        for out_idx in range(self.out_features):
            probs = []
            for distr_idx in range(self.num_distr):
                sigma = torch.log(1 + torch.exp(self.centers[out_idx][distr_idx][1]))
                estimate = (x - self.centers[out_idx][distr_idx][0]) ** 2 / (2 * sigma * sigma)
                probs.append(self.gaussian_activation(estimate))
            probs = torch.stack(probs, 1)
            outputs.append(torch.argmax(probs, dim=-1))
        outputs = torch.stack(outputs, 1)
        return outputs

    def gaussian_activation(self, x):
        return torch.exp(-torch.sum(x, dim=-1))


"""

class Dynamic_estimator(torch.nn.Module):
    def __init__(self, in_features, out_features=200, num_distr=1):
        super().__init__()
        self.training = True
        self.centers = {}
        self.num_distr = num_distr
        self.in_features = in_features
        self.out_features = out_features
        # self.dense = nn.Linear(in_features,out_features)
        self.rewards = {}
        self.top_reward = 500
        self.threshold = 0.8
        for i in range(out_features):
            self.centers[i] = []
            self.rewards[i] = []
            for idx in range(num_distr):
                mean = torch.nn.Parameter(torch.rand(in_features, requires_grad=True))
                self.register_parameter('mean%d%d' % (idx, i), mean)
                rho = torch.nn.Parameter(torch.rand(in_features, requires_grad=True))
                self.register_parameter('rho%d%d' % (idx, i), rho)
                self.centers[i].append([mean, rho])
                self.rewards[i].append(0)

    def reset_rewards(self):
        for out_idx in range(self.out_features):
            for distr_idx in range(self.num_distr):
                self.rewards[out_idx][distr_idx] = 0

    def forward(self, x):
        outputs = []
        # self.reg = []
        for out_idx in range(self.out_features):
            probs = []
            for distr_idx in range(self.num_distr):
                sigma = torch.log(1 + torch.exp(self.centers[out_idx][distr_idx][1]))
                estimate = (x - self.centers[out_idx][distr_idx][0]) ** 2 / (2 * sigma * sigma)
                P = self.gaussian_activation(estimate)
                if self.rewards[out_idx][distr_idx] < self.top_reward:
                    if distr_idx == 0:
                        self.rewards[out_idx][distr_idx] += 1
                    else:
                        if self.rewards[out_idx][distr_idx-1] < self.top_reward:
                            P = P * 0.0
                        else:
                            w = torch.lt(probs[-1], self.threshold).float()
                            P = P * w
                            self.rewards[out_idx][distr_idx] += 1
                else:
                    w = torch.gt(P, self.threshold).float()
                    P = P * w

                probs.append(P)

            probs = torch.stack(probs, 1)

            ##################
            if self.training:
                # diff = self.num_distr * torch.max(probs,dim=-1)[0] - torch.sum(probs,dim=-1)
                # probs = diff / (diff + self.num_distr - self.num_distr * torch.max(probs, dim=-1, keepdim=False)[0])
                probs = torch.max(probs, dim=-1)[0]
                # self.reg.append(1-(torch.max(probs,dim=-1)[0] * (self.num_distr+1) - torch.sum(probs,
                # dim=-1))/self.num_distr )
            ##################
            else:
                probs = (torch.max(probs, dim=-1)[0] * (self.num_distr + 1) - torch.sum(probs, dim=-1)) / self.num_distr
            outputs.append(probs)
        # self.reg = torch.stack(self.reg,1)
        outputs = torch.stack(outputs, 1)
        return outputs

    # uti
    # agitaion
    # activaties
    # MCM to create the entropy
    # keep updating based on the activaties of the users
    # Find the change point

    def get_distr_index(self, x):
        outputs = []
        for out_idx in range(self.out_features):
            probs = []
            for distr_idx in range(self.num_distr):
                sigma = torch.log(1 + torch.exp(self.centers[out_idx][distr_idx][1]))
                estimate = (x - self.centers[out_idx][distr_idx][0]) ** 2 / (2 * sigma * sigma)
                probs.append(self.gaussian_activation(estimate))
            probs = torch.stack(probs, 1)
            outputs.append(torch.argmax(probs, dim=-1))
        outputs = torch.stack(outputs, 1)
        return outputs

    def gaussian_activation(self, x):
        return torch.exp(-torch.sum(x, dim=-1))

"""
