#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata
"""Graph Neural Network file"""
import numpy as np
import warnings

EPS = 0.001


class GNN:
    """Graph neural network class"""

    def __init__(self, N, D, x):
        """ constructor

        :param N:　# of vetex
        :param D: dimension of x
        :param x: x on node
        :param W: weight matrix
        """

        self.N = N
        self.D = D
        self.x = x

    def get_adjacency_matrix(self, edge):
        """make adjacency matrix
        :param edge: undirected edge list
        :return m: adjacency matrix
        """
        m = [[0] * self.N for i in range(self.N)]
        n_edge = [[edge[i][j] - 1 for j in range(len(edge[i]))] for i in range(len(edge))]
        for path in n_edge:
            m[path[0]][path[1]] = 1
            m[path[1]][path[0]] = 1
        m = np.array(m)
        return m

    @staticmethod
    def ReLU(x):
        y = np.maximum(0, x)
        return y

    def aggregate_1(self, x, adjacency_matrix):
        """Aggregation 1

        :param x:feature vector on vertex
        :param adjacency_matrix:
        :return a:D times N matrix
        """
        assert x.shape == (self.D, self.N), """x dimension mismatch in aggregate1"""
        a = np.zeros((self.D, self.N))
        for v in range(adjacency_matrix.shape[0]):
            for w in range(adjacency_matrix.shape[1]):
                if (adjacency_matrix[v][w] == 1):
                    a[:, v] += x[:, w]
        return a

    def aggregate_2(self, W, a):
        """Aggregation 2

        :param W: Weight matrix
        :param a:output of aggregation 1
        :return: new x
        """
        new_x = np.zeros((self.D, self.N))
        for v in range(self.N):
            new_x[:, v] = self.ReLU(W @ a[:, v])
        return new_x

    def readout(self, x):
        """READOUT

        :return h: feature vector of graph
        """
        assert x.shape == (self.D, self.N), """x dimension mismatch in readout"""
        h = np.zeros((self.D, 1))
        for v in range(self.N):
            h += x[:, v].reshape(self.D, 1)
        return h

    def readout_all(self, W, T, adjacency_matrix):
        """ aggragation and readout

        :param W:weight matrix
        :param T:# step of aggregation
        :return h:
        """
        nx = self.x
        for i in range(T):
            a = self.aggregate_1(nx, adjacency_matrix)
            nx = self.aggregate_2(W, a)
        h = self.readout(nx)
        return h

    @staticmethod
    def sigmoid(x):
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                return 1.0 / (1.0 + np.exp(-x))
            except RuntimeWarning as e:
                return 1.0/-x

    def calc_prob(self, W, A, b, T, adjacency_matrix):
        """ calculate probability

        :param W: weight matrix of graph
        :param A: parameter of classifier R^D vector
        :param b: parameter of classifier R
        :param T:# step of aggregation
        :return p: probability
        """
        assert A.shape == (self.D, 1), """dimension mismatch"""
        s = A.T @ self.readout_all(W, T, adjacency_matrix) + b
        p = self.sigmoid(s)
        assert p.shape == (1, 1), "dimension mismatch in calc_prob"
        return p

    def loss(self, y, p):
        """calculate loss of Graph

        :param y:label
        :param p:probability
        :return:
        """
        assert p.shape == (1, 1), "dimension mismatch in loss"
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                l = -y * np.log(p) - (1 - y) * np.log(1 - p)
            except RuntimeWarning as e:
                l = y * np.log(1 + np.exp(-p)) + (1 - y) * np.log(1 + np.exp(p))
        assert l.shape == (1, 1), "dimension mismatch in loss"
        return l

    def calc_gradient(self, W, A, b, y, T, adjacency_matrix):
        """ calculate gradient

        :param W:initial W
        :param A:initial A
        :param b:initial b
        :param y:label
        :param T: # step of aggregation
        :return:
        """
        p1 = self.calc_prob(W, A, b, T, adjacency_matrix)
        l1 = self.loss(y, p1)
        gradA = np.zeros_like(A)
        for i in range(A.shape[0]):
            onehot = [0 if j != i else 1 for j in range(A.shape[0])]
            onehot = np.array(onehot).reshape(A.shape[0], 1)
            p2 = self.calc_prob(W, A + EPS * onehot, b, T, adjacency_matrix)
            l2 = self.loss(y, p2)
            grad = (l2 - l1) / EPS
            gradA[i] = grad
        p2 = self.calc_prob(W, A, b + EPS, T, adjacency_matrix)
        l2 = self.loss(y, p2)
        gradB = (l2 - l1) / EPS
        tmpW = np.copy(W)
        gradW = np.zeros_like(W)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                onehot = [0 if k != j else 1 for k in range(W.shape[1])]
                onehot = np.array(onehot).reshape(1, W.shape[1])
                tmpW[i, :] = EPS * onehot
                p2 = self.calc_prob(tmpW, A, b, T, adjacency_matrix)
                l2 = self.loss(y, p2)
                grad = (l2 - l1) / EPS
                gradW[i][j] = grad

        grad = {"A": gradA, "b": gradB, "W": gradW, "loss": l1}
        return grad

    def GD(self, alpha, W, A, b, y, T, adjacency_matrix):
        """ calculate gradient

        :param alpha:learing rate
        :param W:initial W
        :param A:initial A
        :param b:initial b
        :param y:label
        :param T:# step of aggregation
        :return:
        """
        p = self.calc_prob(W, A, b, T, adjacency_matrix)
        l = self.loss(y, p)
        itr = 0
        print("iteration:", itr, " ,loss:", l[0][0])
        while l >= 0.01:
            grad = self.calc_gradient(W, A, b, y, T, adjacency_matrix)
            W = W - alpha * grad["W"]
            A = A - alpha * grad["A"]
            b = b - alpha * grad["b"]
            itr += 1
            print("iteration:", itr, " ,loss:", float(grad['loss']))

    def SGD(self, alpha, W, A, b, T, batch):
        """ Stochastic gradient descent on minibatch

        :param alpha:learning rate
        :param W:weight matrix
        :param A:parameter
        :param b:parameter
        :param T:# step of aggregation
        :param batch:list of dict
        :return:
        """
        B = len(batch) # Batch size
        assert B >= 1, "batch size must be >= 1"+str(batch)
        sumW = np.zeros_like(W)
        sumA = np.zeros_like(A)
        sumb = 0
        loss = 0
        for i in range(B):
            grad = self.calc_gradient(W, A, b, batch[i]['label'], T, batch[i]['adjacency_matrix'])
            sumW += grad["W"]
            sumA += grad["A"]
            sumb += grad["b"]
            loss += grad["loss"]
        W = W - alpha/B * sumW
        A = A - alpha/B * sumA
        b = b - alpha/B * sumb
        return {"W": W, "A": A, "b": b, "loss": loss}