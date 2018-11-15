from netflows import Graph
import numpy as np
from scipy.io import loadmat

from netflows.funcs import WElinearsolve, WEaffinesolve, WEbprsolve
from netflows.funcs import SOlinearsolve, SOaffinesolve, SObprsolve

import pickle

with open('celegans_null.pickle', 'rb') as f:
    celegans = pickle.load(f)

#celegans_adj1 = celegans['celegans_adj_1']
celegans_adj2 = celegans['celegans_adj_2_rand']
celegans_dist = celegans['celegans_dist']

# G_celegans1 = Graph(adj=celegans_adj1, dist=celegans_dist, weights=celegans_adj1)
#
# for row in range(G_celegans1.adj.shape[0]):
#     for col in range(G_celegans1.adj.shape[1]):
#         if row == col:
#             continue
#         else:
#             print('now computing the WE flow of node pair (%d, %d)' % (row, col))
#             WElinearsolve(G_celegans1, row, col, tol=1e-7, maximum_iter=100000, cutoff=None)
#             #WEaffinesolve(G_celegans1, row, col, tol=1e-7, maximum_iter=100000, cutoff=None)

G_celegans2 = Graph(adj=celegans_adj2, dist=celegans_dist, weights=celegans_adj2)

for row in range(G_celegans2.adj.shape[0]):
    for col in range(G_celegans2.adj.shape[1]):
        if row == col:
            continue
        else:
            print('now computing the WE flow of node pair (%d, %d)' % (row, col))
            WElinearsolve(G_celegans2, row, col, tol=1e-7, maximum_iter=100000, cutoff=None)
            WEaffinesolve(G_celegans2, row, col, tol=1e-7, maximum_iter=100000, cutoff=None)

with open('G_celegans_rand.pickle', 'wb') as f:
    pickle.dump({'G_celegans2_rand':G_celegans2}, f)