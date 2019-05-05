#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata
"""Graph Neural Network file"""
import numpy as np
import warnings
from joblib import Parallel, delayed

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
                if adjacency_matrix[v][w] == 1:
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
                return 1.0 / -x

    def calc_prob(self, param, T, adjacency_matrix):
        """ calculate probability

        :param param:
        :param T:# step of aggregation
        :return p: probability
        """
        assert param["A"].shape == (self.D, 1), """dimension mismatch"""
        s = param["A"].T @ self.readout_all(param["W"], T, adjacency_matrix) + param["b"]
        p = self.sigmoid(s)
        assert p.shape == (1, 1), "dimension mismatch in calc_prob"
        return p

    def predict(self, param, T, adjacency_matrix):
        """ predict label

        :param param: dict
        :param T:
        :param adjacency_matrix:
        :return:
        """
        p = self.calc_prob(param, T, adjacency_matrix)
        if p > 1 / 2:
            return 1
        else:
            return 0

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

    def calc_gradient(self, param, y, T, adjacency_matrix):
        """ calculate gradient

        :param param:
        :param y:label
        :param T: # step of aggregation
        :return:
        """
        A = param["A"]
        W = param["W"]
        b = param["b"]
        p1 = self.calc_prob(param, T, adjacency_matrix)
        l1 = self.loss(y, p1)
        gradA = np.zeros_like(A)
        for i in range(A.shape[0]):
            onehot = [0 if j != i else 1 for j in range(A.shape[0])]
            onehot = np.array(onehot).reshape(A.shape[0], 1)
            tmp = {"W": W, "A": A + EPS * onehot, "b": b}
            p2 = self.calc_prob(tmp, T, adjacency_matrix)
            l2 = self.loss(y, p2)
            grad = (l2 - l1) / EPS
            gradA[i] = grad
        tmp = {"W": W, "A": A, "b": b + EPS}
        p2 = self.calc_prob(tmp, T, adjacency_matrix)
        l2 = self.loss(y, p2)
        gradB = (l2 - l1) / EPS
        gradW = np.zeros_like(W)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                tmpW = np.copy(W)
                onehot = [0 if k != j else 1 for k in range(W.shape[1])]
                onehot = np.array(onehot).reshape(1, W.shape[1])
                tmpW[i, :] = tmpW[i, :] + EPS * onehot
                tmp = {"W": tmpW, "A": A, "b": b}
                p2 = self.calc_prob(tmp, T, adjacency_matrix)
                l2 = self.loss(y, p2)
                grad = (l2 - l1) / EPS
                gradW[i][j] = grad

        grad = {"A": gradA, "b": gradB, "W": gradW, "loss": l1}
        return grad

    def GD(self, alpha, param, y, T, adjacency_matrix):
        """ calculate gradient

        :param alpha:learing rate
        :param param:dict
        :param y:label
        :param T:# step of aggregation
        :return:
        """
        p = self.calc_prob(param, T, adjacency_matrix)
        l = self.loss(y, p)
        itr = 0
        print("iteration:", itr, " ,loss:", l[0][0])
        while l >= 0.01:
            grad = self.calc_gradient(param, y, T, adjacency_matrix)
            W = param["W"] - alpha * grad["W"]
            A = param["A"] - alpha * grad["A"]
            b = param["b"] - alpha * grad["b"]
            param = {"W": W, "A": A, "b": b}
            itr += 1
            l = grad['loss']
            print("iteration:", itr, " ,loss:", float(grad['loss']))

    def SGD(self, alpha, param, T, batch):
        """ Stochastic gradient descent on minibatch

        :param alpha:learning rate
        :param param:dict
        :param T:# step of aggregation
        :param batch:list of dict
        :return:
        """
        B = len(batch)  # Batch size
        assert B >= 1, "batch size must be >= 1" + str(batch)
        sumW = np.zeros_like(param["W"])
        sumA = np.zeros_like(param["A"])
        sumb = 0
        loss = 0
        res = Parallel(n_jobs=-1)(
            [delayed(self.calc_gradient)(param, batch[i]['label'], T, batch[i]['adjacency_matrix']) for i in range(B)])
        for i in range(B):
            grad = res[i]
            sumW += grad["W"]
            sumA += grad["A"]
            sumb += grad["b"]
            loss += grad["loss"]
        W = param["W"] - alpha / B * sumW
        A = param["A"] - alpha / B * sumA
        b = param["b"] - alpha / B * sumb
        param = {"W": W, "A": A, "b": b}
        return {"param": param, "loss": loss / B}

    def Momentum_SGD(self, alpha, param, T, batch, omega, eta):
        """ Momentum Stochastic gradient descent on minibatch

        :param alpha:learning rate
        :param param:
        :param T:# step of aggregation
        :param batch:list of dict
        :param omega:diff of previous step
        :param eta: mement parameter
        :return:
        """
        B = len(batch)  # Batch size
        assert B >= 1, "batch size must be >= 1" + str(batch)
        sumW = np.zeros_like(param["W"])
        sumA = np.zeros_like(param["A"])
        sumb = 0
        loss = 0
        res = Parallel(n_jobs=-1)(
            [delayed(self.calc_gradient)(param, batch[i]['label'], T, batch[i]['adjacency_matrix']) for i in range(B)])
        for i in range(B):
            grad = res[i]
            sumW += grad["W"]
            sumA += grad["A"]
            sumb += grad["b"]
            loss += grad["loss"]
        omega["W"] = - alpha / B * sumW + eta * omega["W"]
        omega["A"] = - alpha / B * sumA + eta * omega["A"]
        omega["b"] = - alpha / B * sumb + eta * omega["b"]
        param = {"W": param["W"] + omega["W"], "A": param["A"] + omega["A"], "b": param["b"] + omega["b"]}
        return {"param": param, "loss": loss / B, "omega": omega}

    def ADAM(self, alpha, beta1, beta2, param, T, batch):
        """ Adam optimization

        :param alpha:
        :param beta1:
        :param beta2:
        :param param:dict includes m v W A V
        :param T:
        :param batch:
        :return:
        """
        B = len(batch)  # Batch size
        EPSILON = 10 ** (-8)
        assert B >= 1, "batch size must be >= 1" + str(batch)
        gradW = np.zeros_like(param["W"])
        gradA = np.zeros_like(param["A"])
        gradb = 0
        loss = 0
        grad = {"W": gradW, "A": gradA, "b": gradb}
        res = Parallel(n_jobs=-1)(
            [delayed(self.calc_gradient)(param, batch[i]['label'], T, batch[i]['adjacency_matrix']) for i in range(B)])
        for i in range(B):
            grad["W"] += res[i]["W"] / B
            grad["A"] += res[i]["A"] / B
            grad["b"] += res[i]["b"] / B
            loss += res[i]["loss"] / B
        m = {}
        m_hat = {}
        v = {}
        v_hat = {}
        param["step"] = param["step"] + 1
        for i in param["m"]:
            m[i] = beta1 * param["m"][i] + (1 - beta1) * grad[i]
            v[i] = beta2 * param["v"][i] + (1 - beta2) * np.square(grad[i])
            m_hat[i] = m[i] / (1 - beta1 ** param["step"])
            v_hat[i] = v[i] / (1 - beta2 ** param["step"])
            param[i] = param[i] - alpha * m_hat[i] / (np.sqrt(v_hat[i]) + EPSILON)
        param["m"] = m
        param["v"] = v
        return {"param": param, "loss": loss}
