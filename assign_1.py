#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata
"""assignment 1"""
import numpy as np
from gnn import GNN

def main():
    N = 4  # 頂点の数
    D = 4  # dimension
    edge = [[1, 2], [2, 3], [2, 4], [3, 4]]  # Edge
    x = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    W = np.array([
        [0.5, 0.3, 0, 0],
        [0.2, 0.1, 0, -0.1],
        [-0.4, -0.5, 1, 0],
        [-3, 0,  0, 1],
    ])
    g = GNN(N, D, edge, x, W)
    a = g.aggregate_1(x)
    # print(a)
    # print(W @ a)
    nx = g.aggregate_2(W, a)
    # print(nx)
    h = g.readout(nx)
    print(h)


if __name__ == '__main__':
    main()