from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.pyplot as plt
import numpy as np
import random
import PIL

random.seed(2020)
torch.manual_seed(2020)


import os

'''
This code is adapted from two sources:
(i) The official PyTorch MNIST example (https://github.com/pytorch/examples/blob/master/mnist/main.py)
(ii) Starter code from Yisong Yue's CS 155 Course (http://www.yisongyue.com/courses/cs155/2020_winter/)
'''

class fcNet(nn.Module):
    '''
    Design your model with fully connected layers (convolutional layers are not
    allowed here). Initial model is designed to have a poor performance. These
    are the sample units you can try:
        Linear, Dropout, activation layers (ReLU, softmax)
    '''
    def __init__(self):
        # Define the units that you will use in your model
        # Note that this has nothing to do with the order in which operations
        # are applied - that is defined in the forward function below.
        super(fcNet, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=20)
        self.fc2 = nn.Linear(20, 10)
        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Define the sequence of operations your model will apply to an input x
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.relu(x)

        output = F.log_softmax(x, dim=1)
        return output

class ConvNet(nn.Module):
    '''
    Design your model with convolutional layers.
    '''
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(8, 8, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(200, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output

'''
class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(8, 8, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(200, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output
'''


class Net(nn.Module):
    '''
    COmpared with teh convnet, this has a 3rd linear layer and doubles the number of the convolution in the 2nd convolution layer
    '''

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, out_channels=8, kernel_size=(3, 3), stride=1)
        self.conv15 = nn.Conv2d(8, out_channels=16, kernel_size=(3, 3), stride=1)
        #self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)

        #self.dropout1 = nn.Dropout2d(0.5)
        #self.dropout2 = nn.Dropout2d(0.5)

        # follow dimensions:
            # conv1 takes 28 to 26
            # maxpool takes 26 to 13
            # conv2 takes 13 to 11
            # maxpool takes 11 to 5

        #self.fc1 = nn.Linear(16 * 5 * 5, 120)
        #self.fc1 = nn.Linear(32 * 4 * 4, 120)
        self.fc1 = nn.Linear(32 * 22 * 22, 3000)
        self.fc15 = nn.Linear(3000, 600)
        self.fc16 = nn.Linear(600, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout1 = nn.Dropout2d(0.3)
        self.dropout2 = nn.Dropout2d(0.3)


    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        #x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv15(x)
        x = F.relu(x)
        x = self.dropout2(x)
        #x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        #x = F.max_pool2d(x, 2)

        size = x.size()[1:]
        dims = 1
        for s in size:
            dims *= s
        x = x.view(-1, dims)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc15(x)
        x = F.relu(x)

        x = self.fc16(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)


        output = F.log_softmax(x, dim=1)
        return output


class Net2(nn.Module):
    '''
    COmpared with the convnet, this has a 3rd linear layer and doubles the number of the convolution in the 2nd convolution layer
    '''

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, out_channels=8, kernel_size=(3, 3), stride=1)
        self.conv15 = nn.Conv2d(8, out_channels=16, kernel_size=(3, 3), stride=1)
        #self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)

        #self.dropout1 = nn.Dropout2d(0.5)
        #self.dropout2 = nn.Dropout2d(0.5)

        # follow dimensions:
            # conv1 takes 28 to 26
            # maxpool takes 26 to 13
            # conv2 takes 13 to 11
            # maxpool takes 11 to 5

        #self.fc1 = nn.Linear(16 * 5 * 5, 120)
        #self.fc1 = nn.Linear(32 * 4 * 4, 120)
        self.fc1 = nn.Linear(32 * 22 * 22, 3000)
        self.fc15 = nn.Linear(3000, 600)
        self.fc16 = nn.Linear(600, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout1 = nn.Dropout2d(0.2)
        self.dropout2 = nn.Dropout2d(0.2)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        #x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv15(x)
        x = F.relu(x)
        x = self.dropout2(x)
        #x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        #x = F.max_pool2d(x, 2)

        size = x.size()[1:]
        dims = 1
        for s in size:
            dims *= s
        x = x.view(-1, dims)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc15(x)
        x = F.relu(x)

        x = self.fc16(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)


        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()   # Set the model to training mode
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output = model(data)                # Make predictions
        loss = F.nll_loss(output, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))

        total_loss = total_loss + loss.item()
    train_loss = total_loss/batch_idx

    return train_loss


def test(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_num += len(data)

    test_loss /= test_num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, test_num,
        100. * correct / test_num))

    return test_loss


def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate your model on the official test set')
    parser.add_argument('--load-model', type=str,
                        help='model file path')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Evaluate on the official test set
    if args.evaluate:
        assert os.path.exists(args.load_model)

        # Set the test model
        model = fcNet().to(device)
        model.load_state_dict(torch.load(args.load_model))

        test_dataset = datasets.MNIST('../data', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        test(model, device, test_loader)

        return

    # Pytorch has default MNIST dataloader which loads data at each iteration
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                transform=transforms.Compose([       # Data preprocessing
                    transforms.ToTensor(),           # Add data augmentation here
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))

    train_dataset_augmented = datasets.MNIST('../data', train=True, download=True,
                                   transform=transforms.Compose([  # Data preprocessing
                                       #transforms.RandomCrop(28, padding=(1, 1, 1, 1)),
                                       #transforms.RandomRotation(4, resample=PIL.Image.BILINEAR),
                                       #transforms.RandomResizedCrop(28, scale=(0.85, 1.0), ratio=(1, 1),
                                       #                             interpolation=2),
                                       transforms.RandomAffine(8, translate=(.065, .065), scale=(0.80, 1.1),
                                                               resample=PIL.Image.BILINEAR),
                                       transforms.ToTensor(),  # Add data augmentation here
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))

    print(type(train_dataset))
    print(len(train_dataset), type(train_dataset[0][0]), type(train_dataset[0][1]), type(train_dataset[0]))

    print("the int is: ", train_dataset[2][1])
    print(np.shape(train_dataset[0][0][0].numpy()))

    idx = [[] for i in range(10)]
    #each row of indexes is a list of indexes in the train_dataset
    #e.g. row 5 containes a list of indexes for the places in train_dataset with images of 5
    print(idx[4])
    for i, img in enumerate(train_dataset):

        #if False:
        if i < 5:
            fig = plt.figure()
            plt.imshow(img[0][0].numpy(), cmap='gray')

            fig = plt.figure()
            plt.imshow(train_dataset_augmented[i][0][0].numpy(), cmap='gray')

        for number in range(10):
            if img[1] == number:
                idx[number].append(i)


    val_idx = [[] for i in range(10)]
    train_idx = [[] for i in range(10)]
    #print(idx[0][1:100])

    for i, number_indx in enumerate(idx):
        random.shuffle(number_indx)
        l = len(number_indx)
        idx_lim = int(l*0.15)
        val_idx[i] = number_indx[0:idx_lim]
        train_idx[i] = number_indx[idx_lim:]


    subset_indices_train = [j for sub in train_idx for j in sub]
    subset_indices_valid = [j for sub in val_idx for j in sub]



    # You can assign indices for training/validation or use a random subset for
    # training by using SubsetRandomSampler. Right now the train and validation
    # sets are built from the same indices - this is bad! Change it so that
    # the training and validation sets are disjoint and have the correct relative sizes.
    #subset_indices_train = range(len(train_dataset))
    #subset_indices_valid = range(len(train_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset_augmented, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(subset_indices_train)
    )
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.test_batch_size,
        sampler=SubsetRandomSampler(subset_indices_valid)
    )

    # Load your model [fcNet, ConvNet, Net]
    model = Net().to(device)

    # Try different optimzers here [Adam, SGD, RMSprop]
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)


    # Set your learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    # Training loop
    train_losses = []
    test_losses = []
    x = []
    fig, ax = plt.subplots(1)

    if True:
        for epoch in range(1, args.epochs + 1):
            #train and test each epoch
            train_loss = train(args, model, device, train_loader, optimizer, epoch)
            test_loss = test(model, device, val_loader)
            scheduler.step()    # learning rate scheduler

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            x.append(epoch - 1)
            ax.plot(x, test_losses, label='test_losses', markersize=2)
            ax.plot(x, train_losses, label='train_losses', markersize=2)

            plt.pause(0.05)

            # You may optionally save your model at each epoch here




        if args.save_model:
            torch.save(model.state_dict(), "mnist_model.pt")


if __name__ == '__main__':
    main()
