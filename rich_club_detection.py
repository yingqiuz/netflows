#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
import numpy as np
import pickle
from netflows import Graph
from bct import randmio_dir_connected,randmio_und_connected,rich_club_wu,rich_club_wd


def load_g(model):
    # which model?
    # load G
    filename = 'total/' + model + '_raw.pickle'
    with open(filename, 'rb') as f:
        G = pickle.load(f)

    return G

def rich_club_detection(adj, num_rand_nks, iter=10):
    """
    :param adj:
    :return:
    """
    if np.allclose(adj, adj.T):
        kmax = np.where(adj!=0, 1, 0).sum(axis=0).max()

        rcc = rich_club_wu(adj, klevel=kmax)

        rand_rcc = np.empty((rcc.shape[0], 0))

        for k in range(num_rand_nks):
            rand_adj, _ = randmio_und_connected(adj, iter)
            rcc_tmp = rich_club_wu(rand_adj, klevel=kmax)
            rand_rcc = np.append(rand_rcc, rcc_tmp[:, np.newaxis], axis = 1)

    else:
        kmax = (np.where(adj!=0, 1, 0).sum(axis=0) + np.where(adj!=0, 1, 0).sum(axis=1)).max()

        rcc = rich_club_wd(adj, klevel=kmax)

        rand_rcc = np.empty((rcc.shape[0], 0))

        for k in range(num_rand_nks):
            rand_adj, _ = randmio_dir_connected(adj, iter)
            rcc_tmp = rich_club_wd(rand_adj, klevel=kmax)
            rand_rcc = np.append(rand_rcc, rcc_tmp[:, np.newaxis], axis = 1)

    # test significance of rich club coefficient
    rcc = np.nan_to_num(rcc)
    rand_rcc = np.nan_to_num(rand_rcc)

    p_vals = np.where(rand_rcc>rcc[:, np.newaxis], 1, 0).sum(axis = 1) / num_rand_nks

    rcc = rcc / rand_rcc.mean(axis=1)

    return rcc, p_vals


if __name__ == '__main__':
    model = sys.argv[1]
    num_rand_nks = sys.argv[2]

    G = load_g(model)

    rcc, p_vals = rich_club_detection(G.adj_weights, int(num_rand_nks), iter=5)

    with open('analysis/stats/' + model + '_rc.pickle', 'wb') as f:
        pickle.dump({'rcc':rcc, 'p_vals':p_vals}, f, protocol=pickle.HIGHEST_PROTOCOL)

