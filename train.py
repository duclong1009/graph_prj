from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, visualize_training_process, accuracy, seed, get_args, get_model, train_fn, test_fn, visualize_training_acc
from models import GAT, SpGAT


if __name__ == '__main__':
    # Training settings
    args = get_args()

    # Set seed
    seed(args.seed)

    # Load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset = args.dataset)
    args.features = features
    args.labels = labels

    # Model and optimizer
    model = get_model(args)
    optimizer = optim.Adam(model.parameters(), 
                        lr=args.lr, 
                        weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    features, adj, labels = Variable(features), Variable(adj), Variable(labels)

    t_total = time.time()
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    bad_counter = 0
    best = args.epochs + 1
    best_epoch = 0
    for epoch in range(args.epochs):
        train_loss, val_loss, train_acc, val_acc= train_fn(epoch, model, optimizer, features, adj, labels, idx_train, idx_val, args) 
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        #save model's state
        torch.save(model.state_dict(), '{}.pkl'.format(epoch))
        if val_loss_list[-1] < best:
            best = val_loss_list[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Restore best model
    print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

    # Testing
    test_loss, acc_test = test_fn(model, features, adj, labels, idx_test)
    visualize_training_process(train_loss_list, val_loss_list,best_epoch , args)
    visualize_training_acc(train_acc_list, val_acc_list, best_epoch, args)