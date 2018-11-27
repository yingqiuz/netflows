import sys,os
import numpy as np
import pickle
from netflows import Graph


def load_g(model):
    # which model?
    # load G
    filename = 'total/' + model + '_raw.pickle'
    with open(filename, 'rb') as f:
        G = pickle.load(f)

    return G


def permutation_g(G):
    # permutation
    G.adj_weights[G.adj!=0] = np.random.permutation(G.adj_weights[G.adj!=0])
    G.wiring_cost[G.adj!=0] = np.random.permutation(G.wiring_cost[G.adj!=0])

    G.WEflowsLinear_edge[G.adj!=0] = np.random.permutation(G.WEflowsLinear_edge[G.adj!=0])
    G.WEtimeLinear_edge[G.adj!=0] = np.random.permutation(G.WEtimeLinear_edge[G.adj!=0])
    G.WEtimeLinear_ratio[G.adj!=0] = np.random.permutation(G.WEtimeLinear_ratio[G.adj!=0])

    return G


if __name__ == "__main__":

    model = sys.argv[1]
    perm_time = sys.argv[2]

    G = load_g(model)

    for k in range(int(perm_time)):
        G_perm = permutation_g(G)

        with open('total/' + model + '_perm' + str(k) + '_raw.pickle', 'wb') as f:
            pickle.dump(G_perm, f, protocol=pickle.HIGHEST_PROTOCOL)