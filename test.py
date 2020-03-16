import torch
from utils.data_loader import load_mnist
from models.base import Linear_base_model


trainloader, testloader = load_mnist()
model = Linear_base_model()

model.train(trainloader)

# Load the model e.g. model.load_state_dict(torch.load('./ckp/mnist_PNN_96.92.pt'))
model.test(testloader)
