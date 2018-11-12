from netflows import Graph
import numpy as np
from scipy.io import loadmat

from netflows.funcs import WElinearsolve, WEaffinesolve, WEbprsolve
from netflows.funcs import SOlinearsolve, SOaffinesolve, SObprsolve

# load mouse data
data = loadmat('/Users/yingqiuzheng/Desktop/Research/selfish_routing/data/G.mat')
data = data['G'][0][0]

mouse = data[1][0][0]
mouse_adj = mouse[0]
mouse_dist = mouse[3]

G_mouse = Graph(adj = mouse_adj, dist=mouse_dist, weights=mouse_adj)

for row in range(G_mouse.adj.shape[0]):
    for col in range(G_mouse.adj.shape[1]):
        if row == col:
            continue
        else:
            print('now computing the WE flow of node pair (%d, %d)' % (row, col))
            if G_mouse.findallpaths(row, col, cutoff=3):
                WElinearsolve(G_mouse, row, col, tol=1e-7, cutoff=3, maximum_iter=100000)
                WEaffinesolve(G_mouse, row, col, tol=1e-7, cutoff=3, maximum_iter=100000)
                
import pickle
with open('G_mouse.pickle', 'wb') as f:
    pickle.dump({'G_mouse':G_mouse}, f, protocol=pickle.HIGHEST_PROTOCOL)
    
