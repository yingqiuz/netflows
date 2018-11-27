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


def permutation_g_dir(G):
    # permutation
    G.adj_weights[G.adj!=0] = np.random.permutation(G.adj_weights[G.adj!=0])
    G.wiring_cost[G.adj!=0] = np.random.permutation(G.wiring_cost[G.adj!=0])

    G.WEflowsLinear_edge[G.adj!=0] = np.random.permutation(G.WEflowsLinear_edge[G.adj!=0])
    G.WEtimeLinear_edge[G.adj!=0] = np.random.permutation(G.WEtimeLinear_edge[G.adj!=0])
    G.WEtimeLinear_ratio[G.adj!=0] = np.random.permutation(G.WEtimeLinear_ratio[G.adj!=0])

    return G


def permutation_g_und(G):

    # undirected network
    G.adj = np.tril(G.adj)
    G.adj_weights = np.tril(G.adj_weights)
    G.wiring_cost = np.tril(G.wiring_cost)
    G.WEflowsLinear_edge = np.tril(G.WEflowsLinear_edge)
    G.WEtimeLinear_edge = np.tril(G.WEtimeLinear_edge)
    G.WEtimeLinear_ratio = np.tril(G.WEtimeLinear_ratio)

    # permutation
    G.adj_weights[G.adj != 0] = np.random.permutation(G.adj_weights[G.adj != 0])
    G.wiring_cost[G.adj != 0] = np.random.permutation(G.wiring_cost[G.adj != 0])

    G.WEflowsLinear_edge[G.adj != 0] = np.random.permutation(G.WEflowsLinear_edge[G.adj != 0])
    G.WEtimeLinear_edge[G.adj != 0] = np.random.permutation(G.WEtimeLinear_edge[G.adj != 0])
    G.WEtimeLinear_ratio[G.adj != 0] = np.random.permutation(G.WEtimeLinear_ratio[G.adj != 0])

    # symmetrize
    G.adj = G.adj + G.adj.T
    G.adj_weights = G.adj_weights + G.adj_weights.T
    G.wiring_cost = G.wiring_cost + G.wiring_cost.T
    G.WEflowsLinear_edge = G.WEflowsLinear_edge + G.WEflowsLinear_edge.T
    G.WEtimeLinear_edge = G.WEtimeLinear_edge + G.WEtimeLinear_edge.T
    G.WEtimeLinear_ratio = G.WEtimeLinear_ratio + G.WEtimeLinear_ratio.T

    return G


if __name__ == "__main__":

    model = sys.argv[1]
    perm_time = sys.argv[2]

    G = load_g(model)

    if np.allclose(G.adj, G.adj.T):
        # undirected network
        for k in range(int(perm_time)):
            G_perm = permutation_g_und(G)

            with open('total/' + model + '_perm' + str(k) + '_raw.pickle', 'wb') as f:
                pickle.dump(G_perm, f, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        # directed network
        for k in range(int(perm_time)):
            G_perm = permutation_g_dir(G)

            with open('total/' + model + '_perm' + str(k) + '_raw.pickle', 'wb') as f:
                pickle.dump(G_perm, f, protocol=pickle.HIGHEST_PROTOCOL)