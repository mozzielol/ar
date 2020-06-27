import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from utils.density_layer import PNN, Density_estimator, Dynamic_estimator
from utils.fc_layers import FC_layer_without_w
from configuration import conf
from utils.conv_layers import MNIST_Conv_block, MNIST_Conv_block_pytorch
import os
from abc import ABC, abstractmethod
from tqdm import tqdm
from utils.data_loader import to_gpu

class Base_model(torch.nn.Module, ABC):
    """docstring for Base_model"""

    def __init__(self):
        self.freq = 10
        super(Base_model, self).__init__()
        self.build()

    @abstractmethod
    def build(self):
        raise NotImplementedError

    def train_model(self, trainloader, learned_imgs=None, verbose=1):
        loss_func = nn.CrossEntropyLoss() if conf.layer_type.startswith('FC') else nn.BCELoss()
        optimizer = optim.Adam(self.parameters())
        self.history = {'loss': [], 'test_acc': []}

        for e in range(conf.num_epoch):
            loss_value = 0.0
            enum = tqdm(enumerate(trainloader, 0)) if verbose else enumerate(trainloader, 0)
            for i, data in enum:
                inputs, labels = data
                inputs, labels = to_gpu(inputs), to_gpu(labels)
                inputs = inputs.view(inputs.size()[0], -1) if conf.model_type == 'NN' else inputs
                if learned_imgs is not None:
                    learned_inputs, learned_targets = next(iter(learned_imgs))
                    inputs = torch.cat([inputs, learned_inputs])
                    labels = torch.cat([labels, learned_targets])
                optimizer.zero_grad()

                classification = self(inputs)

                labels = labels if conf.layer_type.startswith('FC') else F.one_hot(labels, conf.output_units).float()
                # l2_norm = 0.
                # for para_idx, param in enumerate(self.last_layer.named_parameters()):
                #     if param[0].startswith('rho'):
                #         l2_norm += torch.norm(torch.log(1 + torch.exp(param[1])))
                loss = loss_func(classification, labels) # + l2_norm
                loss.backward()
                optimizer.step()

                loss_value += loss.item()

                if i % self.freq == 0:
                    loss_value /= self.freq
                    self.history['loss'].append(loss_value)
                    msg = 'Epoch :{} / {}, loss {:.4f}'.format(e + 1, conf.num_epoch, loss_value)
                    # print('Epoch :{} / {}, loss {:.4f}'.format(e, conf.num_epoch, loss_value), end='\r')
                    loss_value = 0
                    if verbose:
                        enum.set_description(msg)
            if verbose:
                print('')

    def dy_train_model(self, trainloader, verbose=1):
        loss_func = nn.CrossEntropyLoss() if conf.layer_type == 'FC' else nn.BCELoss()
        optimizer = optim.Adam(self.parameters())
        self.history = {'loss': [], 'test_acc': []}

        def batch_train(classification, inputs, labels, loss_value, loss_func, optimizer, train_next=False,
                        threshold=0.9):
            if train_next and classification is not None:
                low_confidence_samples = torch.max(classification, axis=1)[0] < threshold
                sample_idx = low_confidence_samples
                sample_for_next_distr = inputs[sample_idx]
                label_for_next_distr = labels[sample_idx]
                for l in np.unique(label_for_next_distr):
                    self.last_layer.set_training_center(l, 1)
                if torch.sum(sample_idx) > 0:
                    classification = self(sample_for_next_distr)
                else:
                    classification = None
                label_for_next_distr = F.one_hot(label_for_next_distr, conf.output_units).float()
                loss = loss_func(classification, label_for_next_distr)
                loss.backward()
                optimizer.step()
                loss_value += loss.item()
                return classification, loss_value, low_confidence_samples, sample_for_next_distr, torch.argmax(
                    label_for_next_distr, 1)

            labels = labels if conf.layer_type == 'FC' else F.one_hot(labels, conf.output_units).float()
            loss = loss_func(classification, labels)
            # loss += 10*(torch.sum(self.last_layer.reg))**2
            loss.backward()
            optimizer.step()
            loss_value += loss.item()

            return loss_value

        for l in range(10):
            self.last_layer.set_training_center(l, 0)
        for e in range(conf.num_epoch):
            next_distr_rate = 0.0
            loss_value = 0.0
            enum = tqdm(enumerate(trainloader, 0)) if verbose else enumerate(trainloader, 0)

            for i, data in enum:
                inputs, labels = data
                inputs = inputs.view(inputs.size()[0], -1) if conf.model_type == 'NN' else inputs
                optimizer.zero_grad()
                for l in range(10):
                    self.last_layer.set_training_center(l, 0)
                classification = self(inputs)

                # Selection Training
                if e > conf.num_epoch / conf.num_distr:
                    # 1.false prediction samples
                    # classification = classification.detach().numpy()
                    # false_prediction_samples = np.argmax(classification, axis=1) != labels
                    # 2. low confidence samples

                    classification, loss_value, low_confidence_samples, sample_for_next_distr, label_for_next_distr = \
                        batch_train(classification, inputs, labels, loss_value, loss_func, optimizer, train_next=True)

                    for t_idx in range(2, conf.num_distr):
                        if e > t_idx * conf.num_epoch // conf.num_distr:
                            classification, loss_value, low_confidence_samples, sample_for_next_distr, label_for_next_distr = \
                                batch_train(classification, sample_for_next_distr, label_for_next_distr, loss_value,
                                            loss_func, optimizer, train_next=True)
                    sample_for_current_distr = inputs
                    label_for_current_distr = labels
                    optimizer.zero_grad()
                    for l in np.unique(label_for_current_distr):
                        self.last_layer.set_training_center(l, 0)
                    if True:  # torch.sum(sample_idx) > 0:
                        classification = self(sample_for_current_distr)
                        loss_value = batch_train(classification, sample_for_current_distr, label_for_current_distr,
                                                 loss_value, loss_func,
                                                 optimizer)

                    next_distr_rate += torch.sum(low_confidence_samples).float() / inputs.shape[0]


                else:
                    loss_value = batch_train(classification, inputs, labels, loss_value, loss_func, optimizer)

                if i % self.freq == 0:
                    loss_value /= self.freq
                    next_distr_rate /= self.freq
                    self.history['loss'].append(loss_value)
                    msg = 'Epoch :{} / {}, loss: {:.4f}, rate: {:.4f}'.format(e + 1, conf.num_epoch, loss_value,
                                                                              next_distr_rate)
                    # print('Epoch :{} / {}, loss {:.4f}'.format(e, conf.num_epoch, loss_value), end='\r')
                    loss_value = 0
                    next_distr_rate = 0
                    if verbose:
                        enum.set_description(msg)
            if verbose:
                print('')

    def test_model(self, testloader, directory=None, save_model=True):
        correct = 0
        total = 0
        try:
            self.last_layer.training = False
        except:
            pass
        precictions = []
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.view(images.size()[0], -1) if conf.model_type == 'NN' else images
                outputs = self(images)
                precictions.append(outputs.data.numpy())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the %d test images: %.4f %%' % (total,
                                                                          100 * correct / total))

        self.history['test_acc'].append(correct / total)

        if save_model:
            path = './ckp/{}/num_distr={}/{}/'.format(directory, conf.num_distr, conf.model_type)
            try:
                os.stat(path)
            except:
                os.makedirs(path)
            torch.save(self.state_dict(),
                       './ckp/{}/num_distr={}/{}/{}_{}.pt'.format(directory, conf.num_distr, conf.model_type,
                                                                  conf.dataset_name,
                                                                  conf.layer_type))
        return correct / total, np.concatenate(precictions)

    def get_distr_index(self, testloader, is_loader=True):
        assert conf.layer_type in ['DE', 'DY'], 'only DE get this function at the moment'
        if is_loader:
            predicted_index = []
            classes = []
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    images = images.view(images.size()[0], -1) if conf.model_type == 'NN' else images
                    outputs = self.index_forward(images)
                    predicted_index.append(outputs[np.arange(images.shape[0]), labels])
                    classes.append(labels)

            return np.concatenate(predicted_index), np.concatenate(classes)
        else:
            images, labels = testloader
            images = torch.from_numpy(images)
            labels = torch.from_numpy(labels).type(torch.int64)
            images = images.view(images.size()[0], -1) if conf.model_type == 'NN' else images
            outputs = self.index_forward(images)
            predicted_index = outputs[np.arange(images.shape[0]), labels]
            classes = labels

            return predicted_index, classes

    def index_forward(self, x):
        for layer in self.layers:
            x = layer(x).clamp(min=0)
        x = self.last_layer.get_distr_index(x)
        return x

    def summarize_rules(self, verbose=1):
        self.last_layer.training = False
        loss_func = nn.BCELoss()
        # summarize all distributions into one
        inputs = torch.randn((10, conf.hidden_units[0]), requires_grad=True, dtype=torch.float)
        # targets = torch.from_numpy(np.random.randint(low=0, high=10, size=(conf.batch_size, 1)))
        targets = torch.from_numpy(np.arange(10))
        optimizer = optim.Adam([inputs])  # default
        best_inputs = None
        # summarize
        for e in range(conf.num_epoch):
            loss_value = 0.0
            for i in range(500):
                optimizer.zero_grad()

                classification = self(inputs)

                labels = targets if conf.layer_type == 'FC' else F.one_hot(targets, conf.output_units).float()
                loss = loss_func(classification, labels)
                loss.backward()
                optimizer.step()

                loss_value += loss.item()

                if i % self.freq == 0:
                    loss_value /= self.freq
                    self.history['loss'].append(loss_value)
                    print('Epoch :{} / {}, loss {:.4f}'.format(e, conf.num_epoch, loss_value), end='\r')
                    loss_value = 0
                    best_inputs = inputs.data.clone()
            if verbose:
                print('')
        return best_inputs, targets


class Linear_base_model(Base_model):
    """docstring for Linear_base_model"""

    def __init__(self):
        super(Linear_base_model, self).__init__()
        assert conf.model_type == 'NN', 'model_type must be NN for linear model, get model_type %s' % conf.model_type
        self.build()

    def build(self):
        self.history = {'loss': [], 'test_acc': []}
        self.layers = torch.nn.ModuleList([])
        for idx in range(len(conf.hidden_units) - 1):
            self.layers.append(torch.nn.Linear(conf.hidden_units[idx], conf.hidden_units[idx + 1]))

        layer_type = conf.layer_type
        assert layer_type in ['PNN', 'DE', 'FC',
                              'DY', 'Sigmoid', 'Sigmoid_no_weights', 'FC_no_weights'], \
            'last layer must be PNN or DE (Density_estimator) or FC (fully-connected) layer '
        if layer_type == 'PNN':
            last_layer = PNN(conf.hidden_units[-1], conf.output_units, num_distr=conf.num_distr)
        elif layer_type == 'DE':
            last_layer = Density_estimator(conf.hidden_units[-1], conf.output_units, num_distr=conf.num_distr)
        elif layer_type == 'DY':
            last_layer = Dynamic_estimator(conf.hidden_units[-1], conf.output_units, num_distr=conf.num_distr)
        elif layer_type == 'FC' or layer_type == 'Sigmoid':
            last_layer = torch.nn.Linear(conf.hidden_units[-1], conf.output_units)
        elif layer_type == 'Sigmoid_no_weights':
            last_layer = FC_layer_without_w(conf.hidden_units[-1], conf.output_units)
        elif layer_type == 'FC_no_weights':
            last_layer = FC_layer_without_w(conf.hidden_units[-1], conf.output_units)

        self.last_layer = last_layer

    def forward(self, x):
        for layer in self.layers:
            x = layer(x).clamp(min=0)

        x = self.last_layer(x)
        if conf.layer_type.startswith('Sigmoid'):
            x = F.sigmoid(x)
        return x


class Convolutional_base_model(Base_model):
    """docstring for Linear_base_model"""

    def __init__(self):
        super(Convolutional_base_model, self).__init__()
        assert conf.model_type == 'CNN', 'model_type must be CNN for CNN model, get model_type %s' % conf.model_type
        self.build()

    def build(self):
        self.history = {'loss': [], 'test_acc': []}
        self.layers = torch.nn.ModuleList([])
        conv_block = MNIST_Conv_block()
        self.layers.append(conv_block)

        hidden_units = np.insert(conf.hidden_units, 0, conv_block.output_dim, axis=0)
        for idx in range(len(hidden_units) - 1):
            self.layers.append(torch.nn.Linear(hidden_units[idx], hidden_units[idx + 1]))

        layer_type = conf.layer_type
        assert layer_type in ['PNN', 'DE',
                              'FC'], 'last layer must be PNN or DE (Density_estimator) or FC (fully-connected) layer '
        if layer_type == 'PNN':
            last_layer = PNN(hidden_units[-1], conf.output_units, num_distr=conf.num_distr)
        elif layer_type == 'DE':
            last_layer = Density_estimator(hidden_units[-1], conf.output_units, num_distr=conf.num_distr)
        elif layer_type == 'FC':
            last_layer = torch.nn.Linear(hidden_units[-1], conf.output_units)

        self.last_layer = last_layer

    def forward(self, x):
        for layer in self.layers:
            x = layer(x).clamp(min=0)

        x = self.last_layer(x)
        return x


class Pytorch_CNN_Model(Base_model):
    """docstring for Linear_base_model"""

    def __init__(self):
        super(Pytorch_CNN_Model, self).__init__()
        assert conf.model_type == 'CNN', 'model_type must be CNN for CNN model, get model_type %s' % conf.model_type
        self.build()

    def build(self):
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, conf.output_units)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
