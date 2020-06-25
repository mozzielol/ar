import torch
import torch.optim as optim
import torch.nn as nn

import os
import numpy as np
from scipy.special import softmax
from models.base import Linear_base_model, Convolutional_base_model
from configuration import conf
import torchvision
from torchvision import transforms
from art.attacks import FastGradientMethod, ProjectedGradientDescent
from art.classifiers import PyTorchClassifier
from art.utils import load_mnist as art_load_mnist

np.random.seed(2020)

def load_mnist_by_category():
    trainset = torchvision.datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    testset = torchvision.datasets.MNIST('../data', train=False, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=conf.batch_size, shuffle=True, num_workers=0)
    return trainloader, testloader

try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle


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
    return np.max(softmax(preds, axis=1), axis=1) if head.startswith('FC') else np.max(preds, axis=1)


# Load pretrained model
HEAD = 'FC_no_weights'
FEATURE = 'NN'
NUM_DISTR = 'num_distr=1'
eps_choices = np.linspace(0.01, 0.3, num=30, endpoint=True)
if HEAD.startswith('FC'):
    filename = 'history/INDEX_{}_{}.pkl'.format(HEAD, NUM_DISTR)
else:
    filename = 'history/INDEX_FC_{}_{}.pkl'.format(HEAD, NUM_DISTR)
history = {}

model_directory = os.path.join('ckp', NUM_DISTR, FEATURE)
conf.num_distr = NUM_DISTR[-1]

if FEATURE == 'CNN':
    conf.model_type, conf.hidden_units = 'CNN', '100'
else:
    conf.model_type, conf.hidden_units = 'NN', '784,200,200'


conf.layer_type = HEAD
if HEAD.startswith('FC'):
    CHECKPOINT = os.path.join(model_directory, 'mnist_FC.pt')
    criterion = nn.CrossEntropyLoss()
else:
    CHECKPOINT = os.path.join(model_directory, 'Sigmoid_mnist_FC.pt')
    criterion = BCEOneHotLoss()

model = Convolutional_base_model() if FEATURE == 'CNN' else Linear_base_model()

# checkpoint = torch.load(CHECKPOINT)
# model.load_state_dict(checkpoint)
# model.last_layer.training = False
trainloader, testloader = load_mnist_by_category()
model.train_model(trainloader, verbose=1)
# Step 1: Load the MNIST dataset
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = art_load_mnist()
# Step 1a: Transpose to N x D format
if type(model) == Linear_base_model:
    x_train = x_train.reshape((x_train.shape[0], -1)).astype(np.float32)
    x_test = x_test.reshape((x_test.shape[0], -1)).astype(np.float32)
else:
    x_train = x_train.transpose((0, 3, 1, 2)).astype(np.float32)
    x_test = x_test.transpose((0, 3, 1, 2)).astype(np.float32)

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
history = {}
# Step 5: Evaluate the ART classifier on benign test examples
predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))
confidence = np.mean(compute_confidence(predictions, HEAD))
history[0] = (accuracy, confidence)
print("Average confidence on benign test examples: %f" % confidence)

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
        confidence = np.mean(compute_confidence(predictions, HEAD))
        print("Average confidence on adversarial test examples: %f" % confidence)

        accs.append(accuracy)
        confs.append(confidence)
        history[eps] = (accuracy, confidence)

    print("eps = %f, mean accuracy on adversarial test examples: %f ~ %f, mean confidence %f" %
          (eps, np.mean(accs), np.std(accs), np.mean(confs)))

save_his()

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
fc = load_his('./history/INDEX_FC_num_distr=1.pkl')
sigmoid = load_his('./history/INDEX_FC_Sigmoid_num_distr=1.pkl')
fc_no_weights = load_his('history/INDEX_FC_no_weights_num_distr=1.pkl')
sigmoid_no_weights = load_his('history/INDEX_Sigmoid_no_weights_num_distr=1.pkl')
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

sigmoid_res = []
fc_res = []
fc_no_w = []
sigmoid_no_w = []
for eps in eps_choices:
    sigmoid_res.append(sigmoid[eps][0])
    fc_res.append(fc[eps][0])
    fc_no_w.append(fc_no_weights[eps][0])
    sigmoid_no_w.append(sigmoid_no_weights[eps][0])
ax1.plot(fc_res, label='softmax + CE')
ax1.plot(sigmoid_res, label='sigmoid + BCE')
ax1.plot(fc_no_w, label='softmax + CE + unit wegihts')
ax1.plot(sigmoid_no_w, label='sigmoid + BCE + unit wegihts')
# plt.xlim(eps_choices[0], eps_choices[-1])
ticks = []
for i in range(len(eps_choices)):
    if i % 5 ==0:
        ticks.append(str(eps_choices[i])[:4])
    else:
        ticks.append(None)
plt.xticks(np.arange(len(eps_choices)), ticks)
ax1.set_xlabel('Attack strength (eps)')
ax1.set_ylabel('Accuracy')
plt.legend()

sigmoid_res = []
fc_res = []
fc_no_w = []
sigmoid_no_w = []
for eps in eps_choices:
    sigmoid_res.append(sigmoid[eps][1])
    fc_res.append(fc[eps][1])
    fc_no_w.append(fc_no_weights[eps][1])
    sigmoid_no_w.append(sigmoid_no_weights[eps][1])
ax2.plot(fc_res, label='softmax + CE')
ax2.plot(sigmoid_res, label='sigmoid + BCE')
ax2.plot(fc_no_w, label='softmax + CE + unit wegihts')
ax2.plot(sigmoid_no_w, label='sigmoid + BCE + unit wegihts')
# plt.xlim(eps_choices[0], eps_choices[-1])
ticks = []
for i in range(len(eps_choices)):
    if i % 5 ==0:
        ticks.append(str(eps_choices[i])[:4])
    else:
        ticks.append(None)
plt.xticks(np.arange(len(eps_choices)), ticks)
ax2.set_xlabel('Attack strength (eps)')
ax2.set_ylabel('Confidence')
plt.legend()
plt.tight_layout()
plt.savefig('./res_plots/FC_and_Sigmoid.png')
for eps in [0.3, .1, .2]:
    print('Old distribution rate : ',
          np.sum(history['ori_idx'][0].numpy() == history['new_idx'+str(eps)][0].numpy()) / y_test.shape[0])
