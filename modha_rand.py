from netflows import Graph
import numpy as np
from scipy.io import loadmat

from netflows.funcs import WElinearsolve, WEaffinesolve, WEbprsolve
from netflows.funcs import SOlinearsolve, SOaffinesolve, SObprsolve

import pickle

with open('modha_null.pickle', 'rb') as f:
    modha = pickle.load(f)

modha_adj = modha['modha_adj_rand']
modha_dist = modha['modha_dist']

G_modha = Graph(adj=modha_adj, dist=modha_dist, weights=modha_adj)

for row in range(G_modha.adj.shape[0]):
    for col in range(G_modha.adj.shape[1]):
        if row == col:
            continue
        else:
            print('now computing the WE flow of node pair (%d, %d)' % (row, col))
            WElinearsolve(G_modha, row, col, tol=1e-7, maximum_iter=100000)
            # WEaffinesolve(G_modha, row, col, tol=1e-7, maximum_iter=100000)

with open('G_modha_rand.pickle', 'wb') as f:
    pickle.dump({'G_modha_rand': G_modha}, f)

