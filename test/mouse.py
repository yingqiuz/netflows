from netflows import Graph
import numpy as np
from scipy.io import loadmat

from netflows.funcs import WElinearsolve, WEaffinesolve, WEbprsolve
from netflows.funcs import SOlinearsolve, SOaffinesolve, SObprsolve

# load mouse data
import pickle
with open('mouse_conn.pickle', 'rb') as f:
    mouse = pickle.load(f)
    
mouse_adj = mouse['mouse_adj']
mouse_dist = mouse['mouse_dist']

G_mouse = Graph(adj = mouse_adj, dist=mouse_dist, weights=mouse_adj)

for row in range(G_mouse.adj.shape[0]):
    for col in range(G_mouse.adj.shape[1]):
        if row == col:
            continue
        else:
            print('now computing the WE flow of node pair (%d, %d)' % (row, col))
            WElinearsolve(G_mouse, row, col, tol=1e-7, cutoff=None, maximum_iter=100000)
            #WEaffinesolve(G_mouse, row, col, tol=1e-7, cutoff=None, maximum_iter=100000)
                
with open('G_mouse.pickle', 'wb') as f:
    pickle.dump({'G_mouse':G_mouse}, f, protocol=pickle.HIGHEST_PROTOCOL)
    
