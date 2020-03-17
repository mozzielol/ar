import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
from scipy.special import softmax
from models.base import Linear_base_model, Convolutional_base_model
from configuration import conf

from art.attacks import FastGradientMethod, ProjectedGradientDescent
from art.classifiers import PyTorchClassifier
from art.utils import load_mnist as art_load_mnist

np.random.seed(2020)


class BCEOneHotLoss(nn.BCELoss):
    def forward(self, input, target):
        return super(BCEOneHotLoss, self).forward(input, F.one_hot(target, 10).float())


def compute_confidence(preds, head='FC'):
    return np.max(softmax(preds, axis=1), axis=1) if head == 'FC' else np.max(preds, axis=1)


# Load pretrained model
HEAD = 'PNN'
FEATURE = 'NN'
NUM_DISTR = 'num_distr=1'

model_directory = os.path.join('ckp', NUM_DISTR, FEATURE)
conf.num_distr = NUM_DISTR[-1]

if HEAD == 'DE':
    CHECKPOINT = os.path.join(model_directory, 'mnist_DE.pt')
    conf.layer_type = 'DE'
    criterion = BCEOneHotLoss()
elif HEAD == 'PNN':
    CHECKPOINT = os.path.join(model_directory, 'mnist_PNN.pt')
    conf.layer_type = 'PNN'
    criterion = BCEOneHotLoss()
else:
    conf.layer_type = 'FC'
    CHECKPOINT = os.path.join(model_directory, 'mnist_FC.pt')
    criterion = nn.CrossEntropyLoss()

model = Convolutional_base_model() if FEATURE == 'CNN' else Linear_base_model()

checkpoint = torch.load(CHECKPOINT)
model.load_state_dict(checkpoint)

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
    input_shape=(784, ) if type(model) == Linear_base_model else (1, 28, 28),
    nb_classes=10,
    preprocessing=(0.1307, 0.3081)
)

# Step 5: Evaluate the ART classifier on benign test examples
predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))
confidence = np.mean(compute_confidence(predictions, HEAD))
print("Average confidence on benign test examples: %f" % confidence)

# FGSM
for eps in np.linspace(.3, .01, 30):
    accs, confs = [], []
    for repeat in range(1):
        print("eps = %f, run %d" % (eps, repeat))
        # Step 6: Generate adversarial test examples
        attack = FastGradientMethod(classifier=classifier, eps=eps, eps_step=eps/3)
        x_test_adv = attack.generate(x=x_test)

        # Step 7: Evaluate the ART classifier on adversarial test examples
        predictions = classifier.predict(x_test_adv)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
        confidence = np.mean(compute_confidence(predictions, HEAD))
        print("Average confidence on adversarial test examples: %f" % confidence)

        accs.append(accuracy)
        confs.append(confidence)

    print("eps = %f, mean accuracy on adversarial test examples: %f ~ %f, mean confidence %f" %
          (eps, np.mean(accs), np.std(accs), np.mean(confs)))

# PGD
# for eps in [0.2, 0.1]:
#     for max_iter in [10, 20, 30, 40, 50]:
#         accs = []
#         for repeat in range(1):
#             print("eps = %f, step = %d, run %d" % (eps, max_iter, repeat))
#             # Step 6: Generate adversarial test examples
#             attack = ProjectedGradientDescent(classifier=classifier, eps=eps, eps_step=eps/3, max_iter=max_iter)
#             x_test_adv = attack.generate(x=x_test)
#
#             # Step 7: Evaluate the ART classifier on adversarial test examples
#             predictions = classifier.predict(x_test_adv)
#             accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
#             print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
#
#             accs.append(accuracy)
#
#         print("eps = %f, step = %d, mean accuracy on adversarial test examples: %f ~ %f" %
#               (eps, max_iter, np.mean(accs), np.std(accs)))
