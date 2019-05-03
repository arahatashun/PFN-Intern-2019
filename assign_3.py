#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata

"""assignment 3"""
import numpy as np
from gnn import GNN
import random
import copy
train_path = '../datasets/train/'
NUM_TRAIN = 2000
random.seed(1)
np.random.seed(1)

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
    for i in range(NUM_TRAIN):
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


def sgd(gnn, batchsize, train_data, alpha, W, A, b, T, epochs):
    """Stochastic Gradient descent

    :param gnn:gnn object
    :param batchsize:batchsie
    :param alpha:learning rate
    :param train_data:train_data
    :return:
    """
    num_train = len(train_data)
    tmp_train = copy.deepcopy(train_data)
    # print(num_train, batchsize)
    loss = 0
    for epoch in range(epochs):
        for i in range(int(num_train/batchsize)-1):
            mini_batch = sampling(batchsize, tmp_train)
            res = gnn.SGD(alpha, W, A, b, T, mini_batch)
            W = res["W"]
            A = res["A"]
            b = res["b"]
            loss += res["loss"]
            # print("iteration:",i+1,"loss:",res["loss"])
        print("epoch:", epoch+1, ", loss:", loss)
        tmp_train = copy.deepcopy(train_data)
    return res


def make_initial():
    D = 8
    N = 15
    x = np.random.normal(0, 0.4, D * N).reshape(D, N)
    A = np.random.normal(0, 0.4, D).reshape(D, 1)
    W = np.random.normal(0, 0.4, D * D).reshape(D, D)
    b = 0
    return {"x": x, "A": A, "W": W, "b": b}

def main():
    train_data = read_train()
    random.shuffle(train_data)
    ini = make_initial()
    gnn = GNN(15, 8, ini['x'])
    sgd(gnn, 20, train_data, 0.001, ini["W"], ini["A"], ini["b"], 2, 10)



if __name__ == '__main__':
    main()