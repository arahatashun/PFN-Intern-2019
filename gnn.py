#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata
"""Graph Neural Network file"""
import numpy as np
import warnings
EPS = 0.001

class GNN:
    """Graph neural network class"""
    def __init__(self, N, D, edge, x, W):
        """ constructor

        :param N:　# of vetex
        :param D: dimension of x
        :param edge: undirected edge list
        :param x: x on node
        :param W: weight matrix
        """

        self.N = N
        self.D = D
        self.edge = edge
        self.adjacency_matrix = self.get_adjacency_matrix()
        self.x = x
        self.W = W

    def get_adjacency_matrix(self):
        """make adjacency matrix
        :return m: adjacency matrix
        """
        m = [[0] * self.N for i in range(self.N)]
        edge = [[self.edge[i][j] - 1 for j in range(len(self.edge[i]))] for i in range(len(self.edge))]
        for path in edge:
            m[path[0]][path[1]] = 1
            m[path[1]][path[0]] = 1
        m = np.array(m)
        return m

    @staticmethod
    def ReLU(x):
        y = np.maximum(0, x)
        return y

    def aggregate_1(self, x):
        """Aggregation 1

        :param x:feature vector on vertex
        :return a:D times N matrix
        """
        assert x.shape == (self.D, self.N), """x dimension mismatch in aggregate1"""
        a = np.zeros((self.D, self.N))
        for v in range(self.N):
            for w in range(self.N):
                if (self.adjacency_matrix[v][w] == 1):
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
            new_x[:, v] = self.ReLU(W @ a[:,v])
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

    def readout_all(self, W, T):
        """ aggragation and readout

        :param W:weight matrix
        :param T:# step of aggregation
        :return h:
        """
        for i in range(T):
            a = self.aggregate_1(self.x)
            nx = self.aggregate_2(W, a)
        h = self.readout(nx)
        return h

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def calc_prob(self, W, A, b, T):
        """ calculate probability

        :param W: weight matrix of graph
        :param A: parameter of classifier R^D vector
        :param b: parameter of classifier R
        :param T:# step of aggregation
        :return p: probability
        """
        assert A.shape == (self.D, 1), """dimension mismatch"""
        s = A.T @ self.readout_all(W, T) + b
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
                print(e)
                l = y * np.log(1 + np.exp(-p)) + (1-y) * np.log(1 + np.exp(p))
        assert l.shape == (1, 1), "dimension mismatch in loss"
        return l


    def calc_gradient(self, W, A, b, y, T):
        """ calculate gradient

        :param W:initial W
        :param A:initial A
        :param b:initial b
        :param y:label
        :param T: # step of aggregation
        :return:
        """
        p1 = self.calc_prob(W, A, b, T)
        l1 = self.loss(y, p1)
        gradA = np.zeros_like(A)
        for i in range(A.shape[0]):
            onehot = [0 if j !=i else 1 for j in range(A.shape[0])]
            onehot = np.array(onehot).reshape(A.shape[0], 1)
            p2 = self.calc_prob(W, A + EPS * onehot, b, T)
            l2 = self.loss(y, p2)
            grad = (l2-l1)/EPS
            gradA[i] = grad
        p2 = self.calc_prob(W, A, b + EPS, T)
        l2 = self.loss(y, p2)
        gradB = (l2 - l1) / EPS
        tmpW = np.copy(W)
        gradW = np.zeros_like(W)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                onehot = [0 if k != j else 1 for k in range(W.shape[1])]
                onehot = np.array(onehot).reshape(1, W.shape[1])
                tmpW[i,:] = EPS * onehot
                p2 = self.calc_prob(tmpW, A, b, T)
                l2 = self.loss(y, p2)
                grad = (l2 - l1) / EPS
                gradW[i][j] = grad

        grad = {"A": gradA, "b": gradB, "W": gradW}
        return grad

    def GD(self, alpha, W, A, b, y, T):
        """ calculate gradient

        :param alpha:learing rate
        :param W:initial W
        :param A:initial A
        :param b:initial b
        :param y:label
        :param T:# step of aggregation
        :return:
        """
        p = self.calc_prob(W, A, b, T)
        l = self.loss(y, p)
        itr = 0
        print("iteration:", itr, " ,loss:", l[0][0])
        while(l>=0.01):
             grad = self.calc_gradient(W, A, b, y, T)
             W = W - alpha * grad["W"]
             A = A - alpha * grad["A"]
             b = b - alpha * grad["b"]
             p = self.calc_prob(W, A, b, T)
             l = self.loss(y, p)
             itr += 1
             print("iteration:", itr, " ,loss:",l[0][0])