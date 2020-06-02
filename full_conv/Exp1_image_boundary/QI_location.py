#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Location classification w 4QI dataset with 
Resnet18, Bagnet33 and Densenet121 
"""
import numpy as np
import torch
import torchvision

np.random.seed(1988)  # for reproducibility
torch.backends.cudnn.deterministic = True
torch.manual_seed(1988)
torch.cuda.manual_seed_all(1988)

import os 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time
from utils_dataset import four_q_gen
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--arch", type=str, default='resnet18',
                    help="choosing from 'resnet18', 'bagnet33','densenet121'")
parser.add_argument("--init_type", type=str, default='pretrained',
                    help="choosing from 'pretrained', 'stratch','random'")
parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
parser.add_argument("--lr", type=int, default=1e-3, help="learning rate (Default: 1e-3)")
parser.add_argument("--momentum", type=int, default=0.9, help="momentum (Default: 0.9)")
parser.add_argument("--weight_decay", type=int, default=0.00005, help="weight decay (Default: 0.00005)")
parser.add_argument("--nesterov", type=bool, default=True, help="nesterov (Default: True)")
parser.add_argument('--lr_min', metavar='1e-6', default=1e-6, type=float,
                    help='lower bound on learning rate')
parser.add_argument('--patience', metavar='5', default=5, type=int,
                    help='patience for reduce learning rate')

opt = parser.parse_args()
print(opt)

use_gpu = torch.cuda.is_available()

# Generating 4-Quadrant Imagenet dataset

valdir = os.path.join('val')  # imagenet valset directory

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

dataset = torchvision.datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

x_train, y_train = four_q_gen(dataset, set_type=0)
x_val, y_val = four_q_gen(dataset, set_type=1)
x_test, y_test = four_q_gen(dataset, set_type=2)

class FourQDataset(Dataset):

    def __init__(self, Xtrain, y_train, transform):
        self.Xtrain = Xtrain
        self.y_train = y_train
        self.transform = transform

    def __len__(self):
        return len(self.Xtrain)

    def __getitem__(self, idx):
        if self.transform:
            self.Xtrain = self.transform(self.Xtrain)
        sample_crop = {'image': self.Xtrain[idx], 'label': self.y_train[idx]}
        return sample_crop

my_train_set = FourQDataset(x_train, y_train, transform=None)
trainloader = torch.utils.data.DataLoader(my_train_set, batch_size=opt.batch_size,
                                          shuffle=True, num_workers=1)
my_val_set = FourQDataset(x_val, y_val, transform=None)
valloader = torch.utils.data.DataLoader(my_val_set, batch_size=opt.batch_size,
                                        shuffle=True, num_workers=1)
my_test_set = FourQDataset(x_test, y_test, transform=None)
testloader = torch.utils.data.DataLoader(my_test_set, batch_size=opt.batch_size,
                                         shuffle=False, num_workers=1)

################################## Train / Val / Test ###################################
arch_name = opt.arch
init_type = opt.init_type
model_checkpoint = 'imagenet_4Q_' + arch_name + '.pth'
directory_name = 'imagenet_4Q/' + arch_name

if not os.path.isdir(directory_name):
    print("path doesn't exist. Trying to make")
    os.makedirs(directory_name)

list_padding = [(0, 0), (16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512)]

for m in list_padding:

    if arch_name == 'resnet18':

        if init_type == 'pretrained':
            model = models.resnet18(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False

        elif init_type == 'random':  # with frozen conv. weights
            model = models.resnet18(pretrained=False)
            for param in model.parameters():
                param.requires_grad = False

        elif init_type == 'stratch':
            model = models.resnet18(pretrained=False)

        model.maxpool.kernel_size = 2
        model.avgpool = nn.AdaptiveMaxPool2d(1)  # changing average pooling with max pooling
        model.fc = nn.Linear(512, 2, bias=True)  # changing the number of classes as 2

    elif arch_name == 'bagnet33':

        import bagnets.pytorch

        if init_type == 'pretrained':
            model = bagnets.pytorch.bagnet33(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False

        elif init_type == 'random':  # with frozen conv. weights
            model = bagnets.pytorch.bagnet33(pretrained=False)
            for param in model.parameters():
                param.requires_grad = False

        elif init_type == 'stratch':
            model = bagnets.pytorch.bagnet33(pretrained=False)

        model.avgpool = nn.AdaptiveMaxPool2d(1)
        model.fc = nn.Linear(2048, 2, bias=True)

    elif arch_name == 'densenet121':

        if init_type == 'pretrained':
            model = models.densenet121(pretrained=True)
            for param in model.parameters():
                param.requires_grad = False

        elif init_type == 'random':  # with frozen conv. weights
            model = models.densenet121(pretrained=False)
            for param in model.parameters():
                param.requires_grad = False

        elif init_type == 'stratch':
            model = models.densenet121(pretrained=False)

        model.classifier = nn.Linear(1024, 2, bias=True)

        # Generate log file
        with open(directory_name + '/' + init_type + '.csv', 'a') as epoch_log:
            epoch_log.write('Padding, Accuracy \n')

    print('Padding size: ', m)
    if m != (0, 0):  # to keep conv type (Same-Conv) of first conv. layer
        if arch_name == 'densenet121':
            model.features.conv0.padding = m
            print(model.features.conv0)
        else:
            model.conv1.padding = m
            print(model.conv1)

    torch.manual_seed(1988)
    model = model.cuda()

    ''' Loss functions and optimizer '''
    criterion = nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = optim.SGD([{'params': model.parameters()}], lr=lr, momentum=opt.momentum,
                            weight_decay=opt.weight_decay, nesterov=opt.nesterov)
    t1 = time.time()
    print(arch_name, ' training starting...')

    # Training

    flag = True
    best = 50.0
    patience = 0
    for epoch in range(opt.epochs):
        running_loss = 0.0
        counter = 0.0
        size = 0.0
        running_corrects = 0.0
        model.train()
        if flag:
            for i, data in enumerate(trainloader, 0):
                images = data['image']
                labels = data['label']
                images = images.type(torch.FloatTensor)
                labels = labels.view(images.shape[0])
                labels = labels.type(torch.LongTensor)

                if use_gpu:
                    images = Variable(images.cuda())
                    labels = Variable(labels.cuda())
                else:
                    images, labels = Variable(images), Variable(labels)

                optimizer.zero_grad()
                outputs = model(images)

                _, predicted = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(predicted == labels)
                counter += 1
                size += labels.shape[0]

            epoch_loss = running_loss / (counter)
            epoch_acc = running_corrects.item() / (size)
            print('Epoch:{:4d}'.format(epoch + 1))
            print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

            # Validation
    
            running_loss_val = 0.0
            correct = 0.0
            total = 0.0
            model.eval()
            for data in valloader:
                images = data['image']
                labels = data['label']
                images = images.type(torch.FloatTensor)
                labels = labels.view(images.shape[0])
                labels = labels.type(torch.LongTensor)

                if use_gpu:
                    images = Variable(images.cuda())
                    labels = Variable(labels.cuda())
                else:
                    images, labels = Variable(images), Variable(labels)

                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss_val += loss.item()
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += torch.sum(predicted == labels)
            running_loss_val /= total
            print('val loss: ', running_loss_val)
            print('Accuracy of the network on the validation images: %.4f %%' % (
                100 * correct.item() / (total * 1)))

            if best > running_loss_val:
                best = running_loss_val
                model_model_wts = model.state_dict()
                torch.save(model.state_dict(model_model_wts), model_checkpoint)
                patience = 0

            elif patience < opt.patience:
                patience += 1
                print('patience', patience)

            elif patience == opt.patience:
                lr = lr * 0.1
                patience = 0
                print('learning rate is changing. New lr :', lr)
                for g in optimizer.param_groups:
                    g['lr'] = lr

                # loading the model
                print('The best model is loading...')
                model.load_state_dict(torch.load(model_checkpoint))

            if lr <= opt.lr_min:
                print('Training is finished.')
                flag = False

    print('Training finished. ')
    t2 = time.time()
    train_time = float(t2-t1)/60.0
    print('Training time:', train_time)

    # -----------------------------------------------------------
    model.load_state_dict(torch.load(model_checkpoint))  # using the best model

    t1 = time.time()

    # Testing

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():

        for idx, data in enumerate(testloader, 0):
            images = data['image']
            labels = data['label']
            images = images.type(torch.FloatTensor)
            labels = labels.view(images.shape[0])
            labels = labels.type(torch.LongTensor)

            if use_gpu:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            else:
                images, labels = Variable(images), Variable(labels)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print('Accuracy of the network on the 2000 test images: %.4f %%' % (
        100 * accuracy))

    t2 = time.time()
    test_time = float(t2-t1)/60.0
    print('Testing time:', test_time)

    del model

    # Write logs
    with open(directory_name + '/' + init_type + '.csv', 'a') as epoch_log:
        epoch_log.write('{}, {:.5f}\n'.format(m[0], accuracy))
