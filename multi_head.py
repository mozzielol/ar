import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import numpy as np
from pnn import *
import torch.nn.functional as F



"""
Test on CIFAR 10
"""
batch_size = 128

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


def train(model):

    ce = nn.CrossEntropyLoss()
    bce = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(10): 
        ce_loss = 0.0
        bce_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            classification, likelihood= model(inputs)
            loss_ce = ce(classification, labels)
            loss_bce = bce(likelihood, torch.ones(likelihood.size()))
            loss = loss_ce + loss_bce
            loss.backward(retain_graph=True)
            optimizer.step()

            # print statistics
            ce_loss += loss_ce.item()
            bce_loss += loss_bce.item()
            if i % 200 == 0:    
                print('[%d, %5d] ce loss: %.3f  bce_loss: %.3f' %
                      (epoch + 1, i + 1, ce_loss / 2000, bce_loss / 2000))
                ce_loss = 0.0
                bce_loss = 0.0

    print('Finished Training')    


def test(model):
    correct = 0
    total = 0
    likelihood = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs, probs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            likelihood += probs.sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%, likelihood: %.4f' % (
        100 * correct / total, likelihood / total))


if __name__ == '__main__':
    TCNN = Multi_head([np.arange(10)])
    print(TCNN)
    train(TCNN)
    test(TCNN)




















