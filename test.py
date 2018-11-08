from netflows import Graph
import numpy as np

from netflows.funcs import WElinearsolve, WEaffinesolve, WEbprsolve
from netflows.funcs import SOlinearsolve, SOaffinesolve, SObprsolve

adj = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
weights = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1/2], [0, 0, 0, 0]])
dist = np.array([[0, 0, 1, 0], [0, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0]])
# new graph
G = Graph(adj = adj.T, dist=dist.T, weights=weights.T)
G.findallpaths(0, 3, cutoff=4)

SOaffinesolve(G, 0, 3, maximum_iter=10000, tol=1e-12) # the flows should be 0 0.5 0.5
WEaffinesolve(G, 0, 3, maximum_iter=10000, tol=1e-12) # the flows should be 0.4 0.2 0.4


# test on macaque data
from netflows import Graph
import numpy as np
from scipy.io import loadmat

data = loadmat('/Users/yingqiuzheng/Desktop/Research/selfish_routing/data/G.mat')
data = data['G'][0][0]

macaque_markov = data[3][0][0]
macaque_markov_adj = macaque_markov[0]
macaque_markov_dist = macaque_markov[2]

macaque_modha = data[4][0][0]
macaque_modha_adj = macaque_modha[0]
macaque_modha_dist = macaque_modha[3]

mouse = data[1][0][0]
mouse_adj = mouse[0]
mouse_dist = mouse[3]

celegans = data[5][0][0]
celegans_adj = celegans[0]
celegans_dist = celegans[2]

#%%capture
G = Graph(adj = macaque_markov_adj, dist=macaque_markov_dist, weights=macaque_markov_adj)
#G.findallpaths(0, 3, cutoff=3)
SOaffinesolve(G, 5, 67, tol=1e-8, cutoff=3)