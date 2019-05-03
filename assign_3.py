#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata

"""assignment 3"""
import numpy as np
from gnn import GNN
from operator import itemgetter
import random
train_path = '../datasets/train/'
num_train = 2000
random.seed(1)

def read_data(path,index):
    """ read graph and label and return dict

    :param path:
    :param i: index
    :return: dict
    """
    n = 0
    with open(path+str(index)+'_graph.txt') as f:
        l = f.readlines()
        n = int(l[0])
    # print(n)
    ad_matrix = np.zeros((n, n))
    for i in range(n):
        ad_matrix[i] = np.array(l[i+1].split())
    # print(ad_matrix)
    with open(path+str(index)+'_label.txt') as f:
        l = f.readlines()
        label = int(l[0])
    return {'n': n, 'adjacency_matrix': ad_matrix, 'label': label, 'index': index}

def read_train():
    l = []
    for i in range(num_train):
        l.append(read_data(train_path, i))
    return l

def sampling(B, train):
    """ Random Sampling
    Note: destructive function
    :param B: # of data
    :param train:list of train dict
    :return: minibatch
    """
    assert B >= 1, "change Batch size >= 1"
    res = train[:B]
    del train[:B]
    return res


def main():
    train_data = read_train()
    random.shuffle(train_data)
    # print(len(train_data))
    batch = sampling(3, train_data)
    print(len(train_data))


if __name__ == '__main__':
    main()