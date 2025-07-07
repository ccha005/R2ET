#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import pickle
from cnn import ResNet9
import torchvision.transforms as T


def test(model, data_loader, on_gpu, print_out=False):
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    model.eval()
    for data, target in data_loader:
        if on_gpu:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        _, pred = torch.max(output, 1)    
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        for i in range(target.shape[0]):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
    if print_out:
        for i in range(10):
            if class_total[i] > 0:
                print('Accuracy of %5s: %2d%% (%2d/%2d)' % (
                    classes[i], 100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]), np.sum(class_total[i])))
            else:
                print('Accuracy of %5s: N/A (no training examples)' % (classes[i]))
        
        print('\nAccuracy (Overall): %2d%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))
    return 100. * np.sum(class_correct) / np.sum(class_total)


if __name__ == '__main__':
    RANDOM_SEED = 24
    torch.random.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    data_name = 'cifar10'
    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()
    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = 20
    # percentage of training set to use as validation
    valid_size = 0.2
    n_epochs = 30
    generate_data = False
    
    
    
    # specify the image classes
    # classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
    #            'dog', 'frog', 'horse', 'ship', 'truck']
    
    if generate_data == True:
        # convert data to a normalized torch.FloatTensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])
        # choose the training and test datasets
        train_data = datasets.CIFAR10('data', train=True,
                                      download=False, transform=transform)
        test_data = datasets.CIFAR10('data', train=False,
                                      download=False, transform=transform)
        
        
        # obtain training indices that will be used for validation
        num_train = len(train_data)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        
        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        # prepare data loaders (combine dataset and sampler)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
            sampler=train_sampler, num_workers=num_workers)
        valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
            sampler=valid_sampler, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
            num_workers=num_workers)
        
        
        # construct a subset for explanation
        sample_num = [10] * 10
        exp_data = []
        exp_idx = []
        for idx, data in enumerate(test_data):
            y = data[1]
            if sample_num[y] > 0:
                sample_num[y] -= 1
                exp_idx.append(idx)
                exp_data.append((data[0], data[1]))
        exp_sampler = SubsetRandomSampler(exp_idx)
        exp_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
            sampler=exp_sampler, num_workers=num_workers)
        
        with open('./data/' + data_name + '_portion_data', 'wb') as f:
            pickle.dump({'train': train_loader,
                          'val': valid_loader,
                          'test': test_loader,
                          'exp': exp_data}, f)
    else:
        with open('./data/' + data_name + '_portion_data', 'rb') as f:
            dataset = pickle.load(f)
        train_loader = dataset['train']
        valid_loader = dataset['val']
        test_loader = dataset['test']
    
    
    
        
    # create a complete CNN
    model = ResNet9()
    print(model)
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        model.cuda()    
        
    # specify loss function
    criterion = nn.CrossEntropyLoss()
    # specify optimizer
    optimizer = optim.SGD(model.parameters(), lr=.01)
    
    
    
    #List to store loss to visualize
    train_losslist = []
    valid_acc_min = 0 # track change in validation loss
    
    for epoch in range(1, n_epochs+1):
    
        # keep track of training and validation loss
        train_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for data, target in train_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()*data.size(0)
            

        # calculate average losses
        train_loss = train_loss/len(train_loader.dataset)
        train_losslist.append(train_loss)

        valid_acc = test(model, valid_loader, train_on_gpu)            
        
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation acc: {:.6f}'.format(
            epoch, train_loss, valid_acc))
        
        if valid_acc >= valid_acc_min:
            print('Validation acc increased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_acc_min,
            valid_acc))
            torch.save(model.state_dict(), './model/' + data_name + '_pretrain_model.pth')
            valid_acc_min = valid_acc
    
    model.load_state_dict(torch.load('./model/' + data_name + '_pretrain_model.pth'))

    test_accuracy = test(model, test_loader, train_on_gpu, True)
    
