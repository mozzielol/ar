import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import numpy as np
from scipy.special import softmax
from models.base import Linear_base_model, Convolutional_base_model
from configuration import conf
from torchvision import transforms
from art.attacks import FastGradientMethod, ProjectedGradientDescent
from art.classifiers import PyTorchClassifier
from art.utils import load_mnist as art_load_mnist

np.random.seed(2020)

# TODO
# no weights
try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle


def load_mnist_by_category(num_category=10):
    trainset = torchvision.datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    testset = torchvision.datasets.MNIST('../data', train=False, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

    if 1 < num_category <= 10:
        for dataset in [trainset, testset]:
            indices = dataset.targets < num_category
            dataset.targets = dataset.targets[indices]
            dataset.data = dataset.data[indices, ...]

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=conf.batch_size, shuffle=True, num_workers=0)
    return trainloader, testloader


def save_his():
    with open(filename, 'wb') as outfile:
        pickle.dump(history, outfile, protocol=pickle.HIGHEST_PROTOCOL)


def load_his(file):
    with open(file, 'rb') as outfile:
        return pickle.load(outfile)


class BCEOneHotLoss(nn.BCELoss):
    def forward(self, input, target):
        return super(BCEOneHotLoss, self).forward(input, target.float())


def compute_confidence(preds, head='FC'):
    cls_idx = np.argmax(preds) == FAKE
    preds = preds[cls_idx]
    zero_class = preds[:, 0]
    one_class = preds[:, 1]
    return zero_class, one_class


# Load pretrained model
HEAD = 'Sigmoid'
FEATURE = 'NN'
NUM_DISTR = 'num_distr=1'
NUM_CLASSES = 2
TRUE = 0
FAKE = 1
conf.output_units = str(NUM_CLASSES)
eps_choices = np.linspace(0.01, 0.3, num=30, endpoint=True)
if HEAD == 'FC':
    filename = 'history/INDEX_FC_{}.pkl'.format(NUM_DISTR)
else:
    filename = 'history/INDEX_FC_Sigmoid_{}.pkl'.format(NUM_DISTR)
history = {}

model_directory = os.path.join('ckp', NUM_DISTR, FEATURE)
conf.num_distr = NUM_DISTR[-1]

if FEATURE == 'CNN':
    conf.model_type, conf.hidden_units = 'CNN', '100'
else:
    conf.model_type, conf.hidden_units = 'NN', '784,200,200'

conf.layer_type = HEAD
if HEAD == 'FC':
    CHECKPOINT = os.path.join(model_directory, 'mnist_FC.pt')
    criterion = nn.CrossEntropyLoss()
else:
    CHECKPOINT = os.path.join(model_directory, 'Sigmoid_mnist_FC.pt')
    criterion = BCEOneHotLoss()

model = Convolutional_base_model() if FEATURE == 'CNN' else Linear_base_model()

trainloader, testloader = load_mnist_by_category(2)

model.train_model(trainloader)
model.test_model(testloader, directory='logits')
# Step 1: Load the MNIST dataset
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = art_load_mnist()
# Step 1a: Transpose to N x D format
if type(model) == Linear_base_model:
    x_train = x_train.reshape((x_train.shape[0], -1)).astype(np.float32)
    x_test = x_test.reshape((x_test.shape[0], -1)).astype(np.float32)
else:
    x_train = x_train.transpose((0, 3, 1, 2)).astype(np.float32)
    x_test = x_test.transpose((0, 3, 1, 2)).astype(np.float32)

train_indices = np.where(np.argmax(y_train, axis=1) == TRUE)
test_incides = np.where(np.argmax(y_test, axis=1) == TRUE)
x_train = x_train[train_indices]
y_train = y_train[train_indices]
x_test = x_test[test_incides]
y_test = y_test[test_incides]

# Step 2a: Define the optimizer
optimizer = optim.Adam(model.parameters())

# Step 3: Create the ART classifier
classifier = PyTorchClassifier(
    model=model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(784,) if type(model) == Linear_base_model else (1, 28, 28),
    nb_classes=10,
    preprocessing=(0.1307, 0.3081)
)
history = {'eps': [], 0: [], 1 : []}
# Step 5: Evaluate the ART classifier on benign test examples
predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))
confidence = compute_confidence(predictions, HEAD)
history['eps'] += [0]*len(confidence[0])
history[0] += confidence[0].tolist()
history[1] += confidence[1].tolist()

# FGSM

# history['ori_idx'] = model.get_distr_index([x_test, np.argmax(y_test, axis=1)], is_loader=False)
for eps in eps_choices:
    accs, confs = [], []
    for repeat in range(1):
        print("eps = %f, run %d" % (eps, repeat))
        # Step 6: Generate adversarial test examples
        attack = FastGradientMethod(classifier=classifier, eps=eps, eps_step=eps / 3)
        x_test_adv = attack.generate(x=x_test)
        # history['new_idx'+str(eps)] = model.get_distr_index([x_test_adv, np.argmax(y_test, axis=1)], is_loader=False)
        # Step 7: Evaluate the ART classifier on adversarial test examples
        predictions = classifier.predict(x_test_adv)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
        confidence = compute_confidence(predictions, HEAD)
        history['eps'] += [0] * len(confidence[0])
        history[0] += confidence[0].tolist()
        history[1] += confidence[1].tolist()


save_his()

# import matplotlib.pyplot as plt
# import seaborn as sns; sns.set()
# fc = load_his('./history/INDEX_FC_num_distr=1.pkl')
# sigmoid = load_his('./history/INDEX_FC_Sigmoid_num_distr=1.pkl')
# sigmoid_res = []
# fc_res = []
# for eps in eps_choices:
#     sigmoid_res.append(sigmoid[eps])
#     fc_res.append(fc[eps])
# plt.plot(fc_res, label='softmax + crossentropy')
# plt.plot(sigmoid_res, label='sigmoid + binary-crossentropy')
# # plt.xlim(eps_choices[0], eps_choices[-1])
# ticks = []
# for i in range(len(eps_choices)):
#     if i % 5 ==0:
#         ticks.append(str(eps_choices[i])[:4])
#     else:
#         ticks.append(None)
# plt.xticks(np.arange(len(eps_choices)), ticks)
# plt.xlabel('Attack strength (eps)')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.savefig('./res_plots/FC_and_Sigmoid.png')
# for eps in [0.3, .1, .2]:
#     print('Old distribution rate : ',
#           np.sum(history['ori_idx'][0].numpy() == history['new_idx'+str(eps)][0].numpy()) / y_test.shape[0])
