import torch
from configuration import conf
from utils.data_loader import load_mnist
from models.base import Linear_base_model, Convolutional_base_model
from utils.visualise import *




# Load the model e.g. model.load_state_dict(torch.load('./ckp/num_distr=1/NN/mnist_FC.pt'))

trainloader, testloader = load_mnist()

layer_type = ['DE','PNN']

for t in layer_type:
	conf.layer_type = t
	if conf.model_type == 'CNN':
	    model = Convolutional_base_model()
	elif conf.model_type == 'NN':
	    model = Linear_base_model()

	model.train_model(trainloader)
	print(model)

	model.test_model(testloader)
