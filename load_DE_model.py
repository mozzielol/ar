import torch
from configuration import conf
conf.num_distr = 2
conf.layer_type = 'DE'

from utils.data_loader import load_mnist
from models.base import Linear_base_model, Convolutional_base_model
from utils.visualise import *



trainloader, testloader = load_mnist()

if conf.model_type == 'CNN':
    model = Convolutional_base_model()
elif conf.model_type == 'NN':
    model = Linear_base_model()




model.load_state_dict(torch.load('./ckp/NN/mnist_DE_2.pt'))

