import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from configuration import conf


def load_cifar10():
    """
    Test on CIFAR 10
    """

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=conf.batch_size,
                                              shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=conf.batch_size,
                                             shuffle=False, num_workers=1)

    return trainloader, testloader


def load_mnist():
    trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('../data', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ])),
        batch_size=conf.batch_size, shuffle=True, num_workers=0)

    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=conf.batch_size, shuffle=True, num_workers=0)

    return trainloader, testloader
