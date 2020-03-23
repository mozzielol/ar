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

    def forward(self,x):
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
                self.register_parameter('center%d%d' % (i,idx), params)
                self.centers[i].append(params)

    def forward(self, x):
        outputs = []
        for out_idx in range(self.out_features):
            probs = []
            for distr_idx in range(self.num_distr):
                probs.append(self.gaussian_activation(((x - self.centers[out_idx][distr_idx]) ** 2).sum(dim=-1)))

            probs = torch.stack(probs,1)
            ##################
            if self.training:
                #diff = self.num_distr * torch.max(probs,dim=-1)[0] - torch.sum(probs,dim=-1)
                #probs = diff / (diff + self.num_distr - self.num_distr * torch.max(probs, dim=-1, keepdim=False)[0])
                probs = (torch.max(probs,dim=-1)[0] * (self.num_distr+1) - torch.sum(probs,dim=-1))/self.num_distr 
            ##################
            else:
                probs = torch.sum(probs, dim=-1) / (torch.sum(probs, dim=-1) \
                                                                  + self.num_distr - self.num_distr * torch.max(probs, dim=-1)[0])
            outputs.append(probs)

        outputs =torch.stack(outputs,1)
        return outputs

    def gaussian_activation(self, x, sigma=1.):
        return torch.exp(-x / (2 * sigma * sigma))



class Density_estimator(torch.nn.Module):
    def __init__(self, in_features, out_features=200, num_distr=1):
        super().__init__()
        self.training = True
        self.centers = {}
        self.num_distr = num_distr
        self.in_features = in_features
        self.out_features = out_features
        #self.dense = nn.Linear(in_features,out_features)

        for i in range(out_features):
            self.centers[i] = []
            for idx in range(num_distr):
                mean = torch.nn.Parameter(torch.rand(in_features, requires_grad=True))
                self.register_parameter('mean%d%d' % (idx,i), mean)
                rho = torch.nn.Parameter(torch.rand(in_features, requires_grad=True))
                self.register_parameter('rho%d%d' % (idx,i), rho)

                self.centers[i].append([mean,rho])

    def forward(self, x):
        outputs = []
        #self.reg = []
        for out_idx in range(self.out_features):
            probs = []
            for distr_idx in range(self.num_distr):
                sigma = torch.log(1 + torch.exp(self.centers[out_idx][distr_idx][1]))
                estimate = (x - self.centers[out_idx][distr_idx][0])**2 /( 2*sigma*sigma)
                probs.append(self.gaussian_activation(estimate))


            probs = torch.stack(probs,1)

            ##################
            if self.training:
                #diff = self.num_distr * torch.max(probs,dim=-1)[0] - torch.sum(probs,dim=-1)
                #probs = diff / (diff + self.num_distr - self.num_distr * torch.max(probs, dim=-1, keepdim=False)[0])
                probs = (torch.max(probs,dim=-1)[0] * (self.num_distr+1) - torch.sum(probs,dim=-1))/self.num_distr 
            #self.reg.append(1-(torch.max(probs,dim=-1)[0] * (self.num_distr+1) - torch.sum(probs,dim=-1))/self.num_distr )
            ##################
            else:
                probs = torch.sum(probs, dim=-1) / (torch.sum(probs, dim=-1) \
                                                                  + self.num_distr - self.num_distr * torch.max(probs, dim=-1)[0])
            outputs.append(probs)
        #self.reg = torch.stack(self.reg,1)
        outputs =torch.stack(outputs,1)
        return outputs

    def gaussian_activation(self, x):
        return torch.exp(-torch.sum(x,dim=-1))
















