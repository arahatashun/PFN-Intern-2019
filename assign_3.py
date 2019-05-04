#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata

"""assignment 3"""
import numpy as np
from gnn import GNN
import random
import copy
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

train_path = '../datasets/train/'
NUM_TRAIN = 2000
random.seed(1)
np.random.seed(1)


def read_data(path, index):
    """ read graph and label and return dict

    :param path:
    :param i: index
    :return: dict
    """
    n = 0
    with open(path + str(index) + '_graph.txt') as f:
        l = f.readlines()
        n = int(l[0])
    # print(n)
    ad_matrix = np.zeros((n, n))
    for i in range(n):
        ad_matrix[i] = np.array(l[i + 1].split())
    # print(ad_matrix)
    with open(path + str(index) + '_label.txt') as f:
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


def sgd(gnn, batchsize, train_data, alpha, param, T, epochs, test):
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
    dict_list = {"loss": [], "train": [], "test": []}
    for epoch in range(epochs):
        loss = 0
        for i in range(int(num_train / batchsize) - 1):
            mini_batch = sampling(batchsize, tmp_train)
            res = gnn.SGD(alpha, param, T, mini_batch)
            param = res["param"]
            loss += res["loss"]
            # print("iteration:",i+1,"loss:",res["loss"])
        test_accuracy = check_prediction(test, gnn, param, T)
        train_accuracy = check_prediction(train_data, gnn, param, T)
        print("epoch:", epoch + 1, ", loss:", loss[0][0],
              ", train:", '{:.2g}'.format(train_accuracy), "test:", '{:.2g}'.format(test_accuracy))
        tmp_train = copy.deepcopy(train_data)
        dict_list["loss"].append(loss[0][0])
        dict_list["train"].append(train_accuracy)
        dict_list["test"].append(test_accuracy)
    return dict_list


def momentum_sgd(gnn, batchsize, train_data, alpha, param, T, epochs, eta, test):
    """Momentum sgd

    :param gnn:
    :param batchsize:
    :param train_data:
    :param alpha:
    :param param: dict
    :param T:
    :param epochs:
    :param eta:
    :return:
    """
    num_train = len(train_data)
    tmp_train = copy.deepcopy(train_data)
    # print(num_train, batchsize)
    omega_W = np.zeros_like(param["W"])
    omega_A = np.zeros_like(param["A"])
    omega_b = 0
    omega = {"W": omega_W, "A": omega_A, "b": omega_b}
    dict_list = {"loss": [], "train": [], "test": []}
    for epoch in range(epochs):
        loss = 0
        for i in range(int(num_train / batchsize) - 1):
            mini_batch = sampling(batchsize, tmp_train)
            res = gnn.Momentum_SGD(alpha, param, T, mini_batch, omega, eta)
            param = res["param"]
            loss += res["loss"]
            omega = res["omega"]
            # print("iteration:",i+1,"loss:",res["loss"])
        test_accuracy = check_prediction(test, gnn, param, T)
        train_accuracy = check_prediction(train_data, gnn, param, T)
        print("epoch:", epoch + 1, ", loss:", loss[0][0], ", train:",
              '{:.2g}'.format(train_accuracy), "test:", '{:.2g}'.format(test_accuracy))
        tmp_train = copy.deepcopy(train_data)
        dict_list["loss"].append(loss[0][0])
        dict_list["train"].append(train_accuracy)
        dict_list["test"].append(test_accuracy)
    return dict_list


def check_prediction(test, gnn, param, T):
    """

    :param test:
    :param param:
    :return:
    """
    ntest = len(test)
    pos = 0
    for i in range(ntest):
        label = test[i]['label']
        p = gnn.predict(param, T, test[i]["adjacency_matrix"])
        if p == label:
            pos = pos + 1
    # print(pos / ntest)
    return pos / ntest


def make_initial():
    D = 8
    N = 15
    x = np.array([[1, 0, 0, 0, 0, 0, 0, 0] for i in range(N)]).reshape(D, N)
    A = np.random.normal(0, 0.4, D).reshape(D, 1)
    W = np.random.normal(0, 0.4, D * D).reshape(D, D)
    b = 0
    param = {"A": A, "W": W, "b": b}
    return {"x": x, "param": param}


def plot(sgd, msgd):
    """

    :param sgd:
    :param msgd:
    :return:
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    epochs = len(sgd["loss"])
    x = np.arange(epochs)
    ax1.set_xlabel('epochs')
    ln1 = ax1.plot(x, sgd["loss"], '-r', label="SGD loss")
    ln2 = ax1.plot(x, msgd["loss"], '-b', label="MSGD loss")
    ax1.set_ylabel('loss')
    ax2 = ax1.twinx()
    ln3 = ax2.plot(x, sgd["train"], '-g', label="SGD train ac")
    ln4 = ax2.plot(x, sgd["test"], '-c',label="SGD test ac")
    ln5 = ax2.plot(x, msgd["train"], '-m', label="MSGD train ac")
    ln6 = ax2.plot(x, msgd["test"], '-y', label="MSGD test ac")
    ax2.set_ylabel('accuracy')
    lns = ln1 + ln2 + ln3 + ln4 + ln5 + ln6
    labs = [l.get_label() for l in lns]
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    ax1.legend(lns, labs, bbox_to_anchor=(1.04,1),  loc="upper left")
    plt.savefig("output.pdf", bbox_inches="tight")
    plt.show()


def main():

    train_data = read_train()
    random.shuffle(train_data)
    test = train_data[1500:]
    train_data = train_data[:1500]
    ini = make_initial()
    gnn = GNN(15, 8, ini['x'])
    EPOCHS = 100
    dmsgd = momentum_sgd(gnn, 50, train_data, 0.001, ini["param"], 2, EPOCHS, 0.9, test)
    dsgd = sgd(gnn, 50, train_data, 0.001, ini["param"], 2, EPOCHS, test)
    with open('sgd.pickle', 'wb') as handle:
        pickle.dump(dsgd, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('msgd.pickle', 'wb') as handle:
        pickle.dump(dmsgd, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('sgd.pickle', 'rb') as handle:
        dsgd = pickle.load(handle)
    with open('msgd.pickle', 'rb') as handle:
        dmsgd = pickle.load(handle)


    plot(dsgd, dmsgd)


if __name__ == '__main__':
    main()
