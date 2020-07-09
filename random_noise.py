import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns;
import torch
import torch.nn as nn
import torchvision
from scipy.special import softmax
from torchvision import transforms
from configuration import conf
from models.base import Linear_base_model, Convolutional_base_model

sns.set()


def add_noise(data, eps=0.5):
    choice = torch.rand(data.shape) < eps
    noise = torch.rand(data.shape)
    noise[choice] = 0
    data = data + noise
    return data


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def load_mnist_by_category(num_category=10, ratio=1.0, noise_eps=0.5):
    trainset = torchvision.datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    testset = torchvision.datasets.MNIST('../data', train=False, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), AddGaussianNoise(0, noise_eps)]))

    if 1 < num_category <= 10:
        for dataset in [trainset, testset]:
            indices = dataset.targets < num_category
            dataset.targets = dataset.targets[indices]
            dataset.data = dataset.data[indices, ...]

    if 0 < ratio < 1.0:
        torch.manual_seed(1234)
        indices = torch.randperm(len(trainset.targets))[:int(round(ratio * len(trainset.targets)))]
        trainset.targets = trainset.targets[indices]
        trainset.data = trainset.data[indices, ...]

    # testset.data = add_noise(testset.data, noise_eps)

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
    conf.output_units = str(NUM_CLASSES)
    filename = './history/random_noise/ratio={}_{}_{}_num_distr={}.pkl'.format(args['ratio'], HEAD, NUM_CLASSES,
                                                                               NUM_DISTR)
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
    trainloader, testloader = load_mnist_by_category(NUM_CLASSES, args['ratio'])
    model.train_model(trainloader, verbose=0)
    history = {'acc': [], 'conf': []}
    for eps in eps_choice:
        trainloader, testloader = load_mnist_by_category(NUM_CLASSES, args['ratio'], eps)
        test_acc, predictions = model.test_model(testloader, directory='random_noise')
        for data in testloader:
            data = data[0]
            plt.imshow(data[0].reshape(28, 28))
            plt.savefig('./res_plots/imgs/gauss_{}.png'.format(eps))
            break
        confidence = np.mean(compute_confidence(predictions, HEAD))
        history['acc'].append(test_acc)
        history['conf'].append(confidence)

    save_his(filename, history)


def get_combinations():
    import itertools
    params = {
        'Head': ['FC', 'PNN', 'DE'],
        'Feature': ['NN'],
        'Num_distr': [3],  # Please fill a single number in this list to plot
        'Num_classes': [10],  # np.arange(2, 11),
        'ratio': [1],  # np.linspace(0.01, 1, num=10, endpoint=True)
    }
    flat = [[(k, v) for v in vs] for k, vs in params.items()]
    return [dict(items) for items in itertools.product(*flat)], params


def run():
    combinations, _ = get_combinations()
    for c_idx, c in enumerate(combinations):
        print('Test model %d/%d' % (c_idx + 1, len(combinations)))
        train_combin(c)


def plot_history():
    combinations, params = get_combinations()
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
    for head in params['Head']:
        filename = './history/random_noise/ratio={}_{}_{}_num_distr={}.pkl'.format(params['ratio'][0], head,
                                                                                   params['Num_classes'][0],
                                                                                   params['Num_distr'][0])
        history = load_his(filename)
        ax1.plot(history['acc'], label=head)
        plt.legend()
        ax1.set_xlabel('Attack strength (eps)')
        ax1.set_ylabel('Accuracy')
        history['conf'].pop(0)
        ax2.plot(history['conf'], label=head)
        ax2.set_xlabel('Attack strength (eps)')
        ax2.set_ylabel('Confidence')
        plt.legend()
    ticks = []
    for i in range(len(eps_choice)):
        if i % 5 == 0 or i == len(eps_choice) - 1:
            ticks.append(str(eps_choice[i])[:4])
        else:
            ticks.append(None)
    plt.xticks(np.arange(len(eps_choice)), ticks)
    plt.tight_layout()
    plt.savefig('res_plots/random_noise/random_noise.png')
    plt.show()


if __name__ == '__main__':
    eps_choice = np.linspace(0, .3, num=15, endpoint=True)
    # run()
    plot_history()
