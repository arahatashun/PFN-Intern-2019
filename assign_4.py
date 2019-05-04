#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata

"""assignment 4"""
import numpy as np
from gnn import GNN
import random
import copy
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from assign_3 import read_train, make_initial, sampling, check_prediction

train_path = '../datasets/train/'
NUM_TRAIN = 2000
random.seed(1)
np.random.seed(1)


def adam(gnn, batchsize, train_data, alpha, beta1, beta2, param, T, epochs, test):
    """adam

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
            res = gnn.ADAM(alpha, beta1, beta2, param, T, mini_batch)
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

def plot(adam, msgd):
    """

    :param adam:
    :param msgd:
    :return:
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    epochs = len(sgd["loss"])
    x = np.arange(epochs)
    ax1.set_xlabel('epochs')
    ln1 = ax1.plot(x, adam["loss"], '-r', label="ADAM loss")
    ln2 = ax1.plot(x, msgd["loss"], '-b', label="MSGD loss")
    ax1.set_ylabel('loss')
    ax2 = ax1.twinx()
    ln3 = ax2.plot(x, adam["train"], '-g', label="ADAM train ac")
    ln4 = ax2.plot(x, adam["test"], '-c', label="ADAM test ac")
    ln5 = ax2.plot(x, msgd["train"], '-m', label="MSGD train ac")
    ln6 = ax2.plot(x, msgd["test"], '-y', label="MSGD test ac")
    ax2.set_ylabel('accuracy')
    lns = ln1 + ln2 + ln3 + ln4 + ln5 + ln6
    labs = [l.get_label() for l in lns]
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    ax1.legend(lns, labs, bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.savefig("adam.pdf", bbox_inches="tight")
    plt.show()

def main():
    train_data = read_train()
    random.shuffle(train_data)
    test = train_data[1500:]
    train_data = train_data[:1500]
    ini = make_initial()
    gnn = GNN(15, 8, ini['x'])
    EPOCHS = 100
    ini["param"]["m"] = {"W": np.zeros_like(ini["param"]["W"]), "A": np.zeros_like(ini["param"]["A"]),
                         "b": np.zeros_like(
                             ini["param"]["b"])}
    ini["param"]["v"] = copy.deepcopy(ini["param"]["m"])
    ini["param"]["step"] = 0
    adgd = adam(gnn, 50, train_data, 0.001, 0.9, 0.999,
                ini["param"], 2, EPOCHS, test)
    with open('adgd.pickle', 'wb') as handle:
        pickle.dump(adgd, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
