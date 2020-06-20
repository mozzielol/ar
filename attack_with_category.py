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
import torchvision
from torchvision import transforms


def load_mnist_by_category(num_category=10):
    trainset = torchvision.datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    testset = torchvision.datasets.MNIST('../data', train=False, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

    if 1 < num_category < 10:
        for dataset in [trainset, testset]:
            indices = dataset.targets < num_category
            dataset.targets = dataset.targets[indices]
            dataset.data = dataset.data[indices, ...]

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=conf.batch_size, shuffle=True, num_workers=0)
        testloader = torch.utils.data.DataLoader(testset, batch_size=conf.batch_size, shuffle=True, num_workers=0)

    return trainloader, testloader


np.random.seed(2020)

try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle


def save_his(filename, history):
    with open(filename, 'wb') as outfile:
        pickle.dump(history, outfile, protocol=pickle.HIGHEST_PROTOCOL)


def load_his(filename):
    with open(filename, 'rb') as outfile:
        return pickle.load(outfile)


class BCEOneHotLoss(nn.BCELoss):
    def forward(self, input, target):
        return super(BCEOneHotLoss, self).forward(input, target.float())


def compute_confidence(preds, head='FC'):
    return np.max(softmax(preds, axis=1), axis=1) if head == 'FC' else np.max(preds, axis=1)


def train_combin(args):
    """
    :param args: dictionary,
        - Head: FC, DE, DY
        - Feature:   CNN or NN
        - Num_distr: number of distribution in DE/DY
        - Num_classes: number of classes to be trained
    :return:
    """
    # Load pretrained model
    HEAD = args['Head']
    FEATURE = args['Feature']
    NUM_DISTR = args['Num_distr']
    NUM_CLASSES = args['Num_classes']

    filename = './history/category/{}_{}_num_distr={}.pkl'.format(HEAD, NUM_CLASSES, NUM_DISTR)
    conf.num_distr = str(NUM_DISTR)

    if FEATURE == 'CNN':
        conf.model_type, conf.hidden_units = 'CNN', '100'
    else:
        conf.model_type, conf.hidden_units = 'NN', '784,200,200'

    if HEAD == 'DE':
        conf.layer_type = 'DE'
        criterion = BCEOneHotLoss()
    elif HEAD == 'PNN':
        conf.layer_type = 'PNN'
        criterion = BCEOneHotLoss()
    elif HEAD == 'DY':
        conf.layer_type = 'DY'
        criterion = BCEOneHotLoss()
    else:
        conf.layer_type = 'FC'
        criterion = nn.CrossEntropyLoss()

    model = Convolutional_base_model() if FEATURE == 'CNN' else Linear_base_model()
    trainloader, testloader = load_mnist_by_category(NUM_CLASSES)
    model.train_model(trainloader, verbose=0)
    model.test_model(testloader, directory='category')
    # Step 1: Load the MNIST dataset
    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = art_load_mnist()
    # Step 1a: Transpose to N x D format
    if type(model) == Linear_base_model:
        x_train = x_train.reshape((x_train.shape[0], -1)).astype(np.float32)
        x_test = x_test.reshape((x_test.shape[0], -1)).astype(np.float32)
    else:
        x_train = x_train.transpose((0, 3, 1, 2)).astype(np.float32)
        x_test = x_test.transpose((0, 3, 1, 2)).astype(np.float32)

    train_indices = np.where(np.argmax(y_train, axis=1) < NUM_CLASSES)
    test_incides = np.where(np.argmax(y_test, axis=1) < NUM_CLASSES)
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
    history = {}
    # Step 5: Evaluate the ART classifier on benign test examples
    predictions = classifier.predict(x_test)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))
    confidence = np.mean(compute_confidence(predictions, HEAD))
    print("Average confidence on benign test examples: %f" % confidence)
    history['initial_acc'] = accuracy
    history['inital_conf'] = confidence
    # FGSM
    for eps in eps_choice:  # np.linspace(.01, .3, 30):
        accs, confs = [], []
        for repeat in range(1):
            print("eps = %f, run %d" % (eps, repeat))
            # Step 6: Generate adversarial test examples
            attack = FastGradientMethod(classifier=classifier, eps=eps, eps_step=eps / 3)
            x_test_adv = attack.generate(x=x_test)
            # Step 7: Evaluate the ART classifier on adversarial test examples
            predictions = classifier.predict(x_test_adv)
            accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
            print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
            confidence = np.mean(compute_confidence(predictions, HEAD))
            print("Average confidence on adversarial test examples: %f" % confidence)

            accs.append(accuracy)
            confs.append(confidence)
            history[eps] = accs

        print("eps = %f, mean accuracy on adversarial test examples: %f ~ %f, mean confidence %f" %
              (eps, np.mean(accs), np.std(accs), np.mean(confs)))

    save_his(filename, history)


def get_combinations():
    import itertools
    params = {
        'Head': ['FC', 'DE'],
        'Feature': ['NN'],
        'Num_distr': [3],  # Please fill a single number in this list to plot
        'Num_classes': np.arange(2, 10),
    }
    flat = [[(k, v) for v in vs] for k, vs in params.items()]
    return [dict(items) for items in itertools.product(*flat)], params


def run():
    combinations, _ = get_combinations()
    for c_idx, c in enumerate(combinations):
        print('Test model %d/%d' % (c_idx + 1, len(combinations)))
        train_combin(c)


def plot_hisotry():
    import matplotlib.pyplot as plt
    import seaborn as sns;
    sns.set()
    _, params = get_combinations()
    history = {}

    for eps in eps_choice:
        for feat in params['Feature']:
            history[feat] = {}
            plt.clf()
            for head in params['Head']:
                history[feat][head] = []
                for num_class in params['Num_classes']:
                    filename = './history/category/{}_{}_num_distr={}.pkl'.format(head, num_class,
                                                                                  params['Num_distr'][0])
                    history[feat][head].append(load_his(filename)[eps])
                plt.plot(history[feat][head], label=head)
            plt.xticks(np.arange(len(params['Num_classes'])), params['Num_classes'])
            plt.xlabel('Number of Classes')
            plt.ylabel('Accuracy')
            title = 'Eps: {}, Architecture {}'.format(eps, params['Feature'])
            plt.title(title)
            plt.legend()
            plt.savefig('./res_plots/{}.png'.format(title))

    for feat in params['Feature']:
        for num_class in params['Num_classes']:
            plt.clf()
            for head in params['Head']:
                acc = []
                for eps in eps_choice:
                    filename = './history/category/{}_{}_num_distr={}.pkl'.format(head, num_class,
                                                                                  params['Num_distr'][0])
                    acc.append(load_his(filename)[eps])
                plt.plot(acc, label=head)
            plt.xlabel('Eps')
            plt.ylabel('Accuracy')
            plt.xticks(np.arange(len(eps_choice)), eps_choice)
            title = 'Eps: {}, Architecture {} Num_classes : {}'.format(eps, params['Feature'], num_class)
            plt.title(title)
            plt.legend()
            plt.savefig('./res_plots/{}.png'.format(title))

    return history


if __name__ == '__main__':
    eps_choice = [0.01, .1, .2]
    # run()
    plot_hisotry()