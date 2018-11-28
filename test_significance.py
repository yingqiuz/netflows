#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys,os,errno
import numpy as np
import pickle
import pandas as pd


def create_distribution_nodal(model, num_perm_models):
    """
    :param model: which species
    :param num_perm_models: how many permutations to include
    :return: a N*1 vector of p values
    """
    with open('analysis/' + model + '_dfs.pickle', 'rb') as f:
        dfs = pickle.load(f)

    df_nodal = dfs['df_nodal']
    columns = [item for item in list(df_nodal) if item[-6:] != '_count']
    # initialize a dataframe to store p values
    df_nodal_p = pd.DataFrame(index = df_nodal.index.tolist(), columns=columns)

    # iterate through the columns
    for col in columns:
        null_dist = np.empty((df_nodal.shape[0], 0))

        # load permutation dfs
        for k in range(num_perm_models):
            with open('analysis/' + model + '_perm' + str(k) + '_dfs.pickle', 'rb') as f:
                dfs_perm = pickle.load(f)
            df_nodal_perm = dfs_perm['df_nodal']
            null_dist = np.append(null_dist, df_nodal_perm.as_matrix(columns=[col]), axis = 1)

        p_values = np.where(null_dist > df_nodal.as_matrix(columns=[col]), 1, 0).sum(axis = 1, dtype=np.float32) / num_perm_models
        df_nodal_p[col] = p_values

    return df_nodal_p


def create_distribution_edge(model, num_perm_models):
    """
    :param model:
    :param num_perm_models:
    :param col:
    :return:
    """
    with open('analysis/' + model + '_dfs.pickle', 'rb') as f:
        dfs = pickle.load(f)

    df_edge = dfs['df_edge']
    columns = [item for item in list(df_edge) if item[-6:] != '_count']
    # initialize a dataframe to store p values
    df_edge_p = pd.DataFrame(index=df_edge.index.tolist(), columns=columns)

    # iterate through the columns
    for col in columns:
        null_dist = np.empty((df_edge.shape[0], 0))

        # load permutation dfs
        for k in range(num_perm_models):
            with open('analysis/' + model + '_perm' + str(k) + '_dfs.pickle', 'rb') as f:
                dfs_perm = pickle.load(f)
            df_edge_perm = dfs_perm['df_edge']
            null_dist = np.append(null_dist, df_edge_perm.as_matrix(columns=[col]), axis=1)

        p_values = np.where(null_dist > df_edge.as_matrix(columns=[col]), 1, 0).sum(axis=1, dtype=np.float32) / num_perm_models
        df_edge_p[col] = p_values

    return df_edge_p


if __name__ == "__main__":
    model = sys.argv[1]
    num_perm_models = int(sys.argv[2])

    df_nodal_p = create_distribution_nodal(model, num_perm_models)
    df_edge_p = create_distribution_edge(model, num_perm_models)

    with open('analysis/stats/' + model + '_dfs_p.pickle', 'wb') as f:
        pickle.dump({'df_nodal_p':df_nodal_p, 'df_edge_p':df_edge_p}, f, protocol=pickle.HIGHEST_PROTOCOL)

