from netflows import Graph
import numpy as np
from scipy.io import loadmat

from netflows.funcs import WElinearsolve, WEaffinesolve, WEbprsolve
from netflows.funcs import SOlinearsolve, SOaffinesolve, SObprsolve

# load mouse data
data = loadmat('/Users/yingqiuzheng/Desktop/Research/selfish_routing/data/G.mat')
data = data['G'][0][0]

human033 = data[11][0][0]
human033_adj = human033[0]
human033_dist = human033[2]

G_human033 = Graph(adj = human033_adj, dist=human033_dist, weights=human033_adj)

for row in range(G_human033.adj.shape[0]):
    for col in range(G_human033.adj.shape[1]):
        if row == col:
            continue
        else:
            print('now computing the WE flow of node pair (%d, %d)' % (row, col))
            if G_human033.findallpaths(row, col, cutoff=4):
                WElinearsolve(G_human033, row, col, tol=1e-7, cutoff=4, maximum_iter=100000)
                WEaffinesolve(G_human033, row, col, tol=1e-7, cutoff=4, maximum_iter=100000)
                
import pickle
with open('G_human033.pickle', 'wb') as f:
    pickle.dump({'G_human033':G_human033}, f, protocol=pickle.HIGHEST_PROTOCOL)
    

