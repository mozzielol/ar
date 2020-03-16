from utils.data_loader import load_mnist
from models.base import Linear_base_model


trainloader, testloader = load_mnist()
model = Linear_base_model()

model.train(trainloader)
model.test(testloader)
