import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn

    
    

class FC_layer_without_w(torch.nn.Module):
    def __init__(self,in_features, out_features):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones((out_features,in_features), requires_grad=False))
        self.b = torch.nn.Parameter(torch.zeros(out_features, requires_grad=True))
        self.register_parameter('w' , self.w)
        self.register_parameter('b' , self.b)

    def forward(self,x): 
        x = F.linear(x,self.w,self.b)

        return x
