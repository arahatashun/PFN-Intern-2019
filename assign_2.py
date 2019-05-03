#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata
"""assignment 2"""
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
    y = 1
    g = GNN(N, D, x)
    alpha = 0.001
    A = np.array([0.0, 0.0, 0.0, 0.0]).reshape(4, 1)
    b = 0
    g.GD(alpha, W, A, b, y, 1,g.get_adjacency_matrix(edge))
if __name__ == '__main__':
    main()

# iteration: 103148  ,loss: 0.0099999332089
# を確認