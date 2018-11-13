from netflows import Graph
import numpy as np
from scipy.io import loadmat

from netflows.funcs import WElinearsolve, WEaffinesolve, WEbprsolve
from netflows.funcs import SOlinearsolve, SOaffinesolve, SObprsolve

import pickle
with open('rat_conn.pickle', 'rb') as f:
    rat = pickle.load(f)
    
rat_adj = rat['rat_adj']

G_rat = Graph(adj = rat_adj, dist=np.ones(rat_adj.shape), weights=rat_adj)

for row in range(G_rat.adj.shape[0]):
    for col in range(G_rat.adj.shape[1]):
        if row == col:
            continue
        else:
            print('now computing the WE flow of node pair (%d, %d)' % (row, col))
            WElinearsolve(G_rat, row, col, tol=1e-7, cutoff=4, maximum_iter=100000)
                
                
                
import pickle
with open('G_rat.pickle', 'wb') as f:
    pickle.dump({'G_rat': G_rat}, f, protocol=pickle.HIGHEST_PROTOCOL)