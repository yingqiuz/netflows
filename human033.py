from netflows import Graph
import numpy as np
from scipy.io import loadmat

from netflows.funcs import WElinearsolve, WEaffinesolve, WEbprsolve
from netflows.funcs import SOlinearsolve, SOaffinesolve, SObprsolve

# load mouse data
import pickle
with open('human033_conn.pickle', 'rb') as f:
    human033 = pickle.load(f)
    
human033_adj = human033['human033_adj']
human033_dist = human033['human033_dist']

G_human033 = Graph(adj = human033_adj, dist=human033_dist, weights=human033_adj)

for row in range(G_human033.adj.shape[0]):
    for col in range(G_human033.adj.shape[1]):
        if row == col:
            continue
        else:
            print('now computing the WE flow of node pair (%d, %d)' % (row, col))
            WElinearsolve(G_human033, row, col, tol=1e-7, cutoff=None, maximum_iter=100000)
            WEaffinesolve(G_human033, row, col, tol=1e-7, cutoff=None, maximum_iter=100000)
                
with open('G_human033.pickle', 'wb') as f:
    pickle.dump({'G_human033':G_human033}, f, protocol=pickle.HIGHEST_PROTOCOL)
    

