import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from utils.density_layer import PNN, Density_estimator
from configuration import conf
from utils.conv_layers import MNIST_Conv_block
from abc import ABC, abstractmethod
from tqdm import tqdm

class Base_model(torch.nn.Module, ABC):
    """docstring for Base_model"""
    def __init__(self):
        super(Base_model, self).__init__()
        self.build()

    @abstractmethod
    def build(self):
        raise NotImplementedError

    def train(self, trainloader):
        loss_func = nn.CrossEntropyLoss() if conf.layer_type == 'FC' else nn.BCELoss()
        optimizer = optim.Adam(self.parameters())
        self.history = {'loss':[], 'test_acc':[]}

        for e in range(conf.num_epoch):
            loss_value = 0.0
            enum = tqdm(enumerate(trainloader, 0))
            for i, data in enum:
                inputs, labels = data
                inputs = inputs.view(inputs.size()[0],-1)
                optimizer.zero_grad()

                classification = self(inputs)

                lables = labels if conf.layer_type == 'FC' else F.one_hot(labels, 10).float() 
                loss = loss_func(classification, labels)
                loss.backward()
                optimizer.step()

                loss_value += loss.item()

                if i % 100 == 0:
                    loss_value /= 200
                    self.history['loss'].append(loss_value)
                    msg = 'Epoch :{} / {}, loss {:.4f}'.format(e+1, conf.num_epoch, loss_value)
                    #print('Epoch :{} / {}, loss {:.4f}'.format(e, conf.num_epoch, loss_value), end='\r')
                    loss_value = 0
                    enum.set_description(msg)
            print('')


    def test(self, testloader, save_model=True):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.view(images.size()[0],-1)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the %d test images: %d %%' % (total,
                100 * correct / total))

        self.history['test_acc'].append(correct / total)

        if save_model:
            torch.save(self.state_dict(), './ckp/{}/{}_{}_{}.pt'.format(conf.model_type,conf.dataset_name, conf.layer_type,correct/total * 100))






class Linear_base_model(Base_model):
    """docstring for Linear_base_model"""
    def __init__(self):
        super(Linear_base_model, self).__init__()
        assert conf.model_type == 'NN', 'model_type must be NN for linear model, get model_type %s'%conf.model_type
        self.build()

    def build(self):
        self.history = {'loss':[], 'test_acc':[]}
        self.layers = torch.nn.ModuleList([])
        for idx in range(len(conf.hidden_units) - 1):
            self.layers.append(torch.nn.Linear(conf.hidden_units[idx], conf.hidden_units[idx + 1]))

        layer_type = conf.layer_type
        assert layer_type in ['PNN', 'DE', 'FC'], 'last layer must be PNN or DE (Density_estimator) or FC (fully-connected) layer '
        if layer_type == 'PNN':
            last_layer = PNN(conf.hidden_units[-1], conf.output_units, num_distr=conf.num_distr)
        elif layer_type == 'DE':
            last_layer = Density_estimator(conf.hidden_units[-1], conf.output_units, num_distr=conf.num_distr)
        elif layer_type == 'FC':
            last_layer = torch.nn.Linear(conf.hidden_units[-1], conf.output_units)
        
        self.last_layer = last_layer

    def forward(self, x):
        for layer in self.layers:
            x = layer(x).clamp(min=0)

        x = self.last_layer(x)
        return x





class Convolutional_base_model(Base_model):
    """docstring for Linear_base_model"""
    def __init__(self):
        super(Convolutional_base_model, self).__init__()
        assert conf.model_type == 'CNN', 'model_type must be CNN for CNN model, get model_type %s'%conf.model_type
        self.build()

    def build(self):
        self.history = {'loss':[], 'test_acc':[]}
        self.layers = torch.nn.ModuleList([])
        conv_block = MNIST_Conv_block()
        self.layers.append(conv_block)
        hidden_units = np.insert(conf.hidden_units,0,conv_block.output_dim,axis=0)
        for idx in range(len(hidden_units) - 1):
            self.layers.append(torch.nn.Linear(hidden_units[idx], hidden_units[idx + 1]))

        layer_type = conf.layer_type
        assert layer_type in ['PNN', 'DE', 'FC'], 'last layer must be PNN or DE (Density_estimator) or FC (fully-connected) layer '
        if layer_type == 'PNN':
            last_layer = PNN(hidden_units[-1], conf.output_units, num_distr=conf.num_distr)
        elif layer_type == 'DE':
            last_layer = Density_estimator(hidden_units[-1], conf.output_units, num_distr=conf.num_distr)
        elif layer_type == 'FC':
            last_layer = torch.nn.Linear(hidden_units[-1], conf.output_units)
        
        self.last_layer = last_layer
        print(self)


    def forward(self, x):
        for layer in self.layers:
            x = layer(x).clamp(min=0)

        x = self.last_layer(x)
        return x


























        