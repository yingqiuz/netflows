#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
import pickle
from netflows import Graph


def load_g(model):
    # which model?
    # load G
    filename = 'total/' + model + '_raw.pickle'
    with open(filename, 'rb') as f:
        G = pickle.load(f)

    return G


def nodal_und(G):

    degree = G.adj.sum(axis=0)

    nodal_density = G.adj_weights.sum(axis=0)
    nodal_cost = G.wiring_cost.sum(axis=0)
    nodal_flow = G.WEflowsLinear_edge.sum(axis = 0)
    nodal_in_time = G.WEtimeLinear_ratio.sum(axis = 0)
    nodal_out_time = G.WEtimeLinear_ratio.sum(axis=1)

    nodal_density_normalized = G.adj_weights.sum(axis=0) / degree
    nodal_cost_normalized = G.wiring_cost.sum(axis=0) / degree
    nodal_flow_normalized = G.WEflowsLinear_edge.sum(axis = 0) / degree
    nodal_in_time_normalized = G.WEtimeLinear_ratio.sum(axis = 0) / degree
    nodal_out_time_normalized = G.WEtimeLinear_ratio.sum(axis=1) / degree

    # initialize the lists
    hubs_count = np.empty((0,))
    non_hubs_count = np.empty((0,))

    hubs_density_ratio = np.empty((0,))
    hubs_nodal_cost_ratio = np.empty((0,))
    hubs_flow_ratio = np.empty((0,))
    hubs_nodal_time_ratio_in = np.empty((0,))
    hubs_nodal_time_ratio_out = np.empty((0,))

    hubs_density_ratio_normalized = np.empty((0,))
    hubs_nodal_cost_ratio_normalized = np.empty((0,))
    hubs_flow_ratio_normalized = np.empty((0,))
    hubs_nodal_time_ratio_in_normalized = np.empty((0,))
    hubs_nodal_time_ratio_out_normalized = np.empty((0,))

    non_hubs_density_ratio = np.empty((0,))
    non_hubs_nodal_cost_ratio = np.empty((0,))
    non_hubs_flow_ratio = np.empty((0,))
    non_hubs_nodal_time_ratio_in = np.empty((0,))
    non_hubs_nodal_time_ratio_out = np.empty((0,))

    non_hubs_density_ratio_normalized = np.empty((0,))
    non_hubs_nodal_cost_ratio_normalized = np.empty((0,))
    non_hubs_flow_ratio_normalized = np.empty((0,))
    non_hubs_nodal_time_ratio_in_normalized = np.empty((0,))
    non_hubs_nodal_time_ratio_out_normalized = np.empty((0,))

    for k in range(degree.max()):

        hubs = np.where(degree > k, 1, 0)
        non_hubs = np.where(degree <= k, 1, 0)

        # store degree threshold k
        hubs_count = np.append(hubs_count, hubs.sum())
        non_hubs_count = np.append(non_hubs_count, non_hubs.sum())

        hubs_density_ratio = np.append(hubs_density_ratio,
                                       nodal_density[hubs == 1].sum() / nodal_density.sum() / hubs.sum())

        hubs_nodal_cost_ratio = np.append(hubs_nodal_cost_ratio,
                                          nodal_cost[hubs == 1].sum() / nodal_cost.sum() / hubs.sum())

        hubs_flow_ratio = np.append(hubs_flow_ratio,
                                     nodal_flow[hubs == 1].sum() / G.total_flow / hubs.sum())

        hubs_nodal_time_ratio_in = np.append(hubs_nodal_time_ratio_in,
                                             nodal_in_time[hubs == 1].sum() /G.total_flow / hubs.sum())

        hubs_nodal_time_ratio_out = np.append(hubs_nodal_time_ratio_out,
                                              nodal_out_time[hubs == 1].sum() /G.total_flow/ hubs.sum())

        # normalized by degree
        hubs_density_ratio_normalized = np.append(hubs_density_ratio_normalized,
                                                  nodal_density_normalized[hubs == 1].sum() /
                                                  nodal_density.sum() / hubs.sum())

        hubs_nodal_cost_ratio_normalized = np.append(hubs_nodal_cost_ratio_normalized,
                                                     nodal_cost_normalized[hubs == 1].sum() /
                                                     nodal_cost.sum() / hubs.sum())

        hubs_flow_ratio_normalized = np.append(hubs_flow_ratio_normalized,
                                               nodal_flow_normalized[hubs == 1].sum() /
                                               G.total_flow / hubs.sum())

        hubs_nodal_time_ratio_in_normalized = np.append(hubs_nodal_time_ratio_in_normalized,
                                                        nodal_in_time_normalized[hubs==1].sum()/G.total_flow/hubs.sum())

        hubs_nodal_time_ratio_out_normalized = np.append(hubs_nodal_time_ratio_out_normalized,
                                                         nodal_out_time_normalized[hubs==1].sum()/G.total_flow/hubs.sum())

        ## non_hubs
        non_hubs_density_ratio = np.append(non_hubs_density_ratio,
                                           nodal_density[non_hubs == 1].sum() / nodal_density.sum() / non_hubs.sum())

        non_hubs_nodal_cost_ratio = np.append(non_hubs_nodal_cost_ratio,
                                              nodal_cost[non_hubs == 1].sum() / nodal_cost.sum() / non_hubs.sum())

        non_hubs_flow_ratio = np.append(non_hubs_flow_ratio,
                                        nodal_flow[non_hubs == 1].sum() / G.total_flow / non_hubs.sum())

        non_hubs_nodal_time_ratio_in = np.append(non_hubs_nodal_time_ratio_in,
                                                 nodal_in_time[non_hubs == 1].sum() / G.total_flow / non_hubs.sum())

        non_hubs_nodal_time_ratio_out = np.append(non_hubs_nodal_time_ratio_out,
                                                  nodal_out_time[non_hubs == 1].sum() / G.total_flow / non_hubs.sum())

        # normalized by degree
        non_hubs_density_ratio_normalized = np.append(non_hubs_density_ratio_normalized,
                                                      nodal_density_normalized[non_hubs == 1].sum() /
                                                      nodal_density.sum() / non_hubs.sum())

        non_hubs_nodal_cost_ratio_normalized = np.append(non_hubs_nodal_cost_ratio_normalized,
                                                         nodal_cost_normalized[non_hubs == 1].sum() /
                                                         nodal_cost.sum() / non_hubs.sum())

        non_hubs_flow_ratio_normalized = np.append(non_hubs_flow_ratio_normalized,
                                                   nodal_flow_normalized[non_hubs == 1].sum() /
                                                   G.total_flow / non_hubs.sum())

        non_hubs_nodal_time_ratio_in_normalized = np.append(non_hubs_nodal_time_ratio_in_normalized,
                                                            nodal_in_time_normalized[non_hubs == 1].sum() /
                                                            G.total_flow / non_hubs.sum())

        non_hubs_nodal_time_ratio_out_normalized = np.append(non_hubs_nodal_time_ratio_out_normalized,
                                                             nodal_out_time_normalized[non_hubs == 1].sum() /
                                                             G.total_flow / non_hubs.sum())

    df_nodal = pd.DataFrame(data={'hubs_count':hubs_count,
                                  'non_hubs_count':non_hubs_count,
                                  'hubs_density_ratio':hubs_density_ratio,
                                  'hubs_nodal_cost_ratio':hubs_nodal_cost_ratio,
                                  'hubs_flow_ratio':hubs_flow_ratio,
                                  'hubs_nodal_time_ratio_in':hubs_nodal_time_ratio_in,
                                  'hubs_nodal_time_ratio_out':hubs_nodal_time_ratio_out,
                                  'hubs_density_ratio_normalized':hubs_density_ratio_normalized,
                                  'hubs_nodal_cost_ratio_normalized':hubs_nodal_cost_ratio_normalized,
                                  'hubs_flow_ratio_normalized':hubs_flow_ratio_normalized,
                                  'hubs_nodal_time_ratio_in_normalized':hubs_nodal_time_ratio_in_normalized,
                                  'hubs_nodal_time_ratio_out_normalized':hubs_nodal_time_ratio_out_normalized,
                                  'non_hubs_density_ratio':non_hubs_density_ratio,
                                  'non_hubs_nodal_cost_ratio':non_hubs_nodal_cost_ratio,
                                  'non_hubs_flow_ratio':non_hubs_flow_ratio,
                                  'non_hubs_nodal_time_ratio_in':non_hubs_nodal_time_ratio_in,
                                  'non_hubs_nodal_time_ratio_out':non_hubs_nodal_time_ratio_out,
                                  'non_hubs_density_ratio_normalized':non_hubs_density_ratio_normalized,
                                  'non_hubs_nodal_cost_ratio_normalized':non_hubs_nodal_cost_ratio_normalized,
                                  'non_hubs_flow_ratio_normalized':non_hubs_flow_ratio_normalized,
                                  'non_hubs_nodal_time_ratio_in_normalized':non_hubs_nodal_time_ratio_in_normalized,
                                  'non_hubs_nodal_time_ratio_out_normalized':non_hubs_nodal_time_ratio_out_normalized},
                            index = range(degree.max()))

    return df_nodal


def edge_und(G):

    rich_club_count = np.empty((0,))
    local_count = np.empty((0,))
    non_rich_club_count = np.empty((0,))

    rich_club_edge_density_ratio = np.empty((0,))
    non_rich_club_edge_density_ratio = np.empty((0,))
    local_edge_density_ratio = np.empty((0,))

    rich_club_edge_cost_ratio = np.empty((0,))
    non_rich_club_edge_cost_ratio = np.empty((0,))
    local_edge_cost_ratio = np.empty((0,))

    rich_club_edge_flow_ratio = np.empty((0,))
    non_rich_club_edge_flow_ratio = np.empty((0,))
    local_edge_flow_ratio = np.empty((0,))

    rich_club_edge_time_ratio = np.empty((0,))
    non_rich_club_edge_time_ratio = np.empty((0,))
    local_edge_time_ratio = np.empty((0,))

    degree = G.adj.sum(axis=0)

    for k in range(degree.max()):
        hubs = np.where(degree > k, 1, 0)
        non_hubs = np.where(degree > k, 0, 1)
        rich_club = np.outer(hubs, hubs) * G.adj
        non_rich_club = (1 - rich_club) * G.adj
        local = np.outer(non_hubs, non_hubs) * G.adj

        rich_club_count = np.append(rich_club_count, rich_club.sum())
        local_count = np.append(local_count, local.sum())
        non_rich_club_count = np.append(non_rich_club_count, non_rich_club.sum())

        rich_club_edge_density_ratio = np.append(rich_club_edge_density_ratio,
                                                 G.adj_weights[rich_club == 1].sum() /
                                                 G.adj_weights.sum() / rich_club.sum())

        non_rich_club_edge_density_ratio = np.append(non_rich_club_edge_density_ratio,
                                                     G.adj_weights[non_rich_club == 1].sum() /
                                                     G.adj_weights.sum() / non_rich_club.sum())

        local_edge_density_ratio = np.append(local_edge_density_ratio,
                                             G.adj_weights[local == 1].sum() /
                                             G.adj_weights.sum() / local.sum())

        rich_club_edge_cost_ratio = np.append(rich_club_edge_cost_ratio,
                                              G.wiring_cost[rich_club == 1].sum() /
                                              G.wiring_cost.sum() / rich_club.sum())

        non_rich_club_edge_cost_ratio = np.append(non_rich_club_edge_cost_ratio,
                                                  G.wiring_cost[non_rich_club == 1].sum() /
                                                  G.wiring_cost.sum() / non_rich_club.sum())

        local_edge_cost_ratio = np.append(local_edge_cost_ratio,
                                          G.wiring_cost[local == 1].sum() /
                                          G.wiring_cost.sum() / local.sum())

        rich_club_edge_flow_ratio = np.append(rich_club_edge_flow_ratio,
                                              G.WEflowsLinear_edge[rich_club == 1].sum() /
                                              G.total_flow / rich_club.sum())

        non_rich_club_edge_flow_ratio = np.append(non_rich_club_edge_flow_ratio,
                                                  G.WEflowsLinear_edge[non_rich_club == 1].sum() /
                                                  G.total_flow / non_rich_club.sum())

        local_edge_flow_ratio = np.append(local_edge_flow_ratio,
                                          G.WEflowsLinear_edge[local == 1].sum() /
                                          G.total_flow / local.sum())

        rich_club_edge_time_ratio = np.append(rich_club_edge_time_ratio,
                                              G.WEcostsLinear_ratio[rich_club == 1].sum() /
                                              G.total_flow / rich_club.sum())

        non_rich_club_edge_time_ratio = np.append(non_rich_club_edge_time_ratio,
                                                  G.WEcostsLinear_ratio[non_rich_club == 1].sum() /
                                                  G.total_flow / non_rich_club.sum())

        local_edge_time_ratio = np.append(local_edge_time_ratio, G.WEcostsLinear_ratio[local == 1].sum() /
                                          G.total_flow / local.sum())

    df_edge = pd.DataFrame(data={'rich_club_count':rich_club_count,
                                 'local_count':local_count,
                                 'non_rich_club_count':non_rich_club_count,
                                 'rich_club_edge_density_ratio':rich_club_edge_density_ratio,
                                 'non_rich_club_edge_density_ratio':non_rich_club_edge_density_ratio,
                                 'local_edge_density_ratio':local_edge_density_ratio,
                                 'rich_club_edge_cost_ratio':rich_club_edge_cost_ratio,
                                 'non_rich_club_edge_cost_ratio':non_rich_club_edge_cost_ratio,
                                 'local_edge_cost_ratio':local_edge_cost_ratio,
                                 'rich_club_edge_flow_ratio':rich_club_edge_flow_ratio,
                                 'non_rich_club_edge_flow_ratio':non_rich_club_edge_flow_ratio,
                                 'local_edge_flow_ratio':local_edge_flow_ratio,
                                 'rich_club_edge_time_ratio':rich_club_edge_time_ratio,
                                 'non_rich_club_edge_time_ratio':non_rich_club_edge_time_ratio,
                                 'local_edge_time_ratio':local_edge_time_ratio},
                           index=range(degree.max()))
    return df_edge


def nodal_dir(G):

    in_degree = G.adj.sum(axis=0)
    out_degree = G.adj.sum(axis=1)

    in_density = G.adj_weights.sum(axis=0)
    out_density = G.adj_weights.sum(axis=1)
    in_cost = G.wiring_cost.sum(axis=0)
    out_cost = G.wiring_cost.sum(axis=1)
    in_flow = G.WEflowsLinear_edge.sum(axis=0)
    out_flow = G.WEflowsLinear_edge.sum(axis=1)
    in_time = G.WEtimeLinear_ratio.sum(axis = 0)
    out_time = G.WEtimeLinear_ratio.sum(axis = 1)

    in_density_normalized = G.adj_weights.sum(axis=0) / in_degree
    out_density_normalized = G.adj_weights.sum(axis=1) / out_degree
    in_cost_normalized = G.wiring_cost.sum(axis=0) / in_degree
    out_cost_normalized = G.wiring_cost.sum(axis=1) / out_degree
    in_flow_normalized = G.WEflowsLinear_edge.sum(axis=0) / in_degree
    out_flow_normalized = G.WEflowsLinear_edge.sum(axis=1) / out_degree
    in_time_normalized = G.WEtimeLinear_ratio.sum(axis=0) / in_degree
    out_time_normalized = G.WEtimeLinear_ratio.sum(axis=1) / out_degree

    hubs_density_ratio_in = np.empty((0, ))
    hubs_density_ratio_out = np.empty((0, ))
    hubs_nodal_cost_ratio_in = np.empty((0,))
    hubs_nodal_cost_ratio_out = np.empty((0,))
    hubs_flow_ratio_in = np.empty((0,))
    hubs_flow_ratio_out = np.empty((0,))
    hubs_nodal_time_ratio_in = np.empty((0,))
    hubs_nodal_time_ratio_out = np.empty((0,))

    hubs_density_ratio_in_normalized = np.empty((0, ))
    hubs_density_ratio_out_normalized = np.empty((0, ))
    hubs_nodal_cost_ratio_in_normalized = np.empty((0,))
    hubs_nodal_cost_ratio_out_normalized = np.empty((0,))
    hubs_flow_ratio_in_normalized = np.empty((0,))
    hubs_flow_ratio_out_normalized = np.empty((0,))
    hubs_nodal_time_ratio_in_normalized = np.empty((0,))
    hubs_nodal_time_ratio_out_normalized = np.empty((0,))

    non_hubs_density_ratio_in = np.empty((0,))
    non_hubs_density_ratio_out = np.empty((0,))
    non_hubs_nodal_cost_ratio_in = np.empty((0,))
    non_hubs_nodal_cost_ratio_out = np.empty((0,))
    non_hubs_flow_ratio_in = np.empty((0,))
    non_hubs_flow_ratio_out = np.empty((0,))
    non_hubs_nodal_time_ratio_in = np.empty((0,))
    non_hubs_nodal_time_ratio_out = np.empty((0,))

    non_hubs_density_ratio_in_normalized = np.empty((0,))
    non_hubs_density_ratio_out_normalized = np.empty((0,))
    non_hubs_nodal_cost_ratio_in_normalized = np.empty((0,))
    non_hubs_nodal_cost_ratio_out_normalized = np.empty((0,))
    non_hubs_flow_ratio_in_normalized = np.empty((0,))
    non_hubs_flow_ratio_out_normalized = np.empty((0,))
    non_hubs_nodal_time_ratio_in_normalized = np.empty((0,))
    non_hubs_nodal_time_ratio_out_normalized = np.empty((0,))

    in_hubs_count = np.empty((0,))
    out_hubs_count = np.empty((0,))
    in_non_hubs_count = np.empty((0,))
    out_non_hubs_count = np.empty((0,))

    for k in range(max(in_degree.max(), out_degree.max())):
        # degree threshold k
        in_hubs = np.where(in_degree > k, 1, 0)
        out_hubs = np.where(out_degree > k, 1, 0)

        in_hubs_count = np.append(in_hubs_count, in_hubs.sum())
        out_hubs_count = np.append(out_hubs_count, out_hubs.sum())

        hubs_density_ratio_in = np.append(hubs_density_ratio_in,
                                             in_density[in_hubs == 1].sum() /
                                             in_density.sum() / in_hubs.sum())
        hubs_density_ratio_out = np.append(hubs_density_ratio_out,
                                              out_density[out_hubs == 1].sum() /
                                              out_density.sum() / out_hubs.sum())

        hubs_nodal_cost_ratio_in = np.append(hubs_nodal_cost_ratio_in,
                                          in_cost[in_hubs == 1].sum() /
                                          in_cost.sum() / in_hubs.sum())
        hubs_nodal_cost_ratio_out = np.append(hubs_nodal_cost_ratio_out,
                                           out_cost[out_hubs == 1].sum() /
                                           out_cost.sum() / out_hubs.sum())

        hubs_flow_ratio_in = np.append(hubs_flow_ratio_in,
                                        in_flow[in_hubs == 1].sum() / G.total_flow / in_hubs.sum())
        hubs_flow_ratio_out = np.append(hubs_flow_ratio_out,
                                         out_flow[out_hubs == 1].sum() / G.total_flow / out_hubs.sum())

        hubs_nodal_time_ratio_in = np.append(hubs_nodal_time_ratio_in,
                                             in_time[in_hubs == 1].sum() / G.total_flow / in_hubs.sum())

        hubs_nodal_time_ratio_out = np.append(hubs_nodal_time_ratio_out,
                                              out_time[out_hubs == 1].sum() / G.total_flow / out_hubs.sum())

        hubs_density_ratio_in_normalized = np.append(hubs_density_ratio_in_normalized,
                                                     in_density_normalized[in_hubs == 1].sum() /
                                                     in_density.sum() / in_hubs.sum())

        hubs_density_ratio_out_normalized = np.append(hubs_density_ratio_out_normalized,
                                                      out_density_normalized[out_hubs == 1].sum() /
                                                      out_density.sum() / out_hubs.sum())

        hubs_nodal_cost_ratio_in_normalized = np.append(hubs_nodal_cost_ratio_in_normalized,
                                                        in_cost_normalized[in_hubs == 1].sum() /
                                                        in_cost.sum() / in_hubs.sum())

        hubs_nodal_cost_ratio_out_normalized = np.append(hubs_nodal_cost_ratio_out_normalized,
                                                         out_cost_normalized[out_hubs == 1].sum() /
                                                         out_cost.sum() / out_hubs.sum())

        hubs_flow_ratio_in_normalized = np.append(hubs_flow_ratio_in_normalized,
                                                  in_flow_normalized[in_hubs == 1].sum() /
                                                  G.total_flow / in_hubs.sum())

        hubs_flow_ratio_out_normalized = np.append(hubs_flow_ratio_out_normalized,
                                                   out_flow_normalized[out_hubs == 1].sum() /
                                                   G.total_flow / out_hubs.sum())

        hubs_nodal_time_ratio_in_normalized = np.append(hubs_nodal_time_ratio_in_normalized,
                                                        in_time_normalized[in_hubs == 1].sum() /
                                                        G.total_flow / in_hubs.sum())

        hubs_nodal_time_ratio_out_normalized = np.append(hubs_nodal_time_ratio_out_normalized,
                                                         out_time_normalized[out_hubs == 1].sum() /
                                                         G.total_flow / out_hubs.sum())

        ### non_hubs
        in_non_hubs_count = np.append(in_non_hubs_count, np.where(in_hubs == 0, 1, 0).sum())
        out_non_hubs_count = np.append(out_non_hubs_count, np.where(out_hubs == 0, 1, 0).sum())

        non_hubs_density_ratio_in = np.append(non_hubs_density_ratio_in,
                                              in_density[in_hubs == 0].sum() /
                                              in_density.sum() / np.where(in_hubs == 0, 1, 0).sum())
        non_hubs_density_ratio_out = np.append(non_hubs_density_ratio_out,
                                               out_density[out_hubs == 0].sum() /
                                               out_density.sum() / np.where(out_hubs == 0, 1, 0).sum())

        non_hubs_nodal_cost_ratio_in = np.append(non_hubs_nodal_cost_ratio_in,
                                                 in_cost[in_hubs == 0].sum() /
                                                 in_cost.sum() / np.where(in_hubs == 0, 1, 0).sum())

        non_hubs_nodal_cost_ratio_out = np.append(non_hubs_nodal_cost_ratio_out,
                                                  out_cost[out_hubs == 0].sum() /
                                                  out_cost.sum() / np.where(out_hubs == 0, 1, 0).sum())

        non_hubs_flow_ratio_in = np.append(non_hubs_flow_ratio_in,
                                           in_flow[in_hubs == 0].sum() / G.total_flow / np.where(in_hubs == 0, 1, 0).sum())
        non_hubs_flow_ratio_out = np.append(non_hubs_flow_ratio_out,
                                            out_flow[out_hubs == 0].sum() / G.total_flow / np.where(out_hubs == 0, 1, 0).sum())

        non_hubs_nodal_time_ratio_in = np.append(non_hubs_nodal_time_ratio_in,
                                                 in_time[in_hubs == 0].sum() / G.total_flow / np.where(in_hubs == 0, 1, 0).sum())

        non_hubs_nodal_time_ratio_out = np.append(non_hubs_nodal_time_ratio_out,
                                                  out_time[out_hubs == 0].sum() / G.total_flow / np.where(out_hubs == 0, 1, 0).sum())

        non_hubs_density_ratio_in_normalized = np.append(non_hubs_density_ratio_in_normalized,
                                                         in_density_normalized[in_hubs == 0].sum() /
                                                         in_density.sum() / np.where(in_hubs == 0, 1, 0).sum())

        non_hubs_density_ratio_out_normalized = np.append(non_hubs_density_ratio_out_normalized,
                                                          out_density_normalized[out_hubs == 0].sum() /
                                                          out_density.sum() / np.where(out_hubs == 0, 1, 0).sum())

        non_hubs_nodal_cost_ratio_in_normalized = np.append(non_hubs_nodal_cost_ratio_in_normalized,
                                                            in_cost_normalized[in_hubs == 0].sum() /
                                                            in_cost.sum() / np.where(in_hubs == 0, 1, 0).sum())

        non_hubs_nodal_cost_ratio_out_normalized = np.append(non_hubs_nodal_cost_ratio_out_normalized,
                                                             out_cost_normalized[out_hubs == 0].sum() /
                                                             out_cost.sum() / np.where(out_hubs == 0, 1, 0).sum())

        non_hubs_flow_ratio_in_normalized = np.append(non_hubs_flow_ratio_in_normalized,
                                                      in_flow_normalized[in_hubs == 0].sum() /
                                                      G.total_flow / np.where(in_hubs == 0, 1, 0).sum())

        non_hubs_flow_ratio_out_normalized = np.append(non_hubs_flow_ratio_out_normalized,
                                                       out_flow_normalized[out_hubs == 0].sum() /
                                                       G.total_flow / np.where(out_hubs == 0, 1, 0).sum())

        non_hubs_nodal_time_ratio_in_normalized = np.append(non_hubs_nodal_time_ratio_in_normalized,
                                                            in_time_normalized[in_hubs == 0].sum() /
                                                            G.total_flow / np.where(in_hubs == 0, 1, 0).sum())

        non_hubs_nodal_time_ratio_out_normalized = np.append(non_hubs_nodal_time_ratio_out_normalized,
                                                             out_time_normalized[out_hubs == 0].sum() /
                                                             G.total_flow / np.where(out_hubs == 0, 1, 0).sum())

    df_nodal = pd.DataFrame(data={'in_hubs_count':in_hubs_count,
                                  'out_hubs_count':out_hubs_count,
                                  'hubs_density_ratio_in':hubs_density_ratio_in,
                                  'hubs_density_ratio_out':hubs_density_ratio_out,
                                  'hubs_nodal_cost_ratio_in':hubs_nodal_cost_ratio_in,
                                  'hubs_nodal_cost_ratio_out':hubs_nodal_cost_ratio_out,
                                  'hubs_flow_ratio_in':hubs_flow_ratio_in,
                                  'hubs_flow_ratio_out':hubs_flow_ratio_out,
                                  'hubs_nodal_time_ratio_in':hubs_nodal_time_ratio_in,
                                  'hubs_nodal_time_ratio_out':hubs_nodal_time_ratio_out,
                                  'hubs_nodal_cost_ratio_in_normalized':hubs_nodal_cost_ratio_in_normalized,
                                  'hubs_nodal_cost_ratio_out_normalized':hubs_nodal_cost_ratio_out_normalized,
                                  'hubs_flow_ratio_in_normalized':hubs_flow_ratio_in_normalized,
                                  'hubs_flow_ratio_out_normalized':hubs_flow_ratio_out_normalized,
                                  'hubs_nodal_time_ratio_in_normalized':hubs_nodal_time_ratio_in_normalized,
                                  'hubs_nodal_time_ratio_out_normalized':hubs_nodal_time_ratio_out_normalized,
                                  'in_non_hubs_count':in_non_hubs_count,
                                  'out_non_hubs_count':out_non_hubs_count,
                                  'non_hubs_density_ratio_in': non_hubs_density_ratio_in,
                                  'non_hubs_density_ratio_out': non_hubs_density_ratio_out,
                                  'non_hubs_nodal_cost_ratio_in':non_hubs_nodal_cost_ratio_in,
                                  'non_hubs_nodal_cost_ratio_out':non_hubs_nodal_cost_ratio_out,
                                  'non_hubs_flow_ratio_in':non_hubs_flow_ratio_in,
                                  'non_hubs_flow_ratio_out':non_hubs_flow_ratio_out,
                                  'non_hubs_nodal_time_ratio_in':non_hubs_nodal_time_ratio_in,
                                  'non_hubs_nodal_time_ratio_out':non_hubs_nodal_time_ratio_out,
                                  'non_hubs_nodal_cost_ratio_in_normalized':non_hubs_nodal_cost_ratio_in_normalized,
                                  'non_hubs_nodal_cost_ratio_out_normalized':non_hubs_nodal_cost_ratio_out_normalized,
                                  'non_hubs_flow_ratio_in_normalized':non_hubs_flow_ratio_in_normalized,
                                  'non_hubs_flow_ratio_out_normalized':non_hubs_flow_ratio_out_normalized,
                                  'non_hubs_nodal_time_ratio_in_normalized':non_hubs_nodal_time_ratio_in_normalized,
                                  'non_hubs_nodal_time_ratio_out_normalized':non_hubs_nodal_time_ratio_out_normalized},
                            index=range(max(in_degree.max(), out_degree.max())))
    return df_nodal


def edge_dir(G):

    rich_club_edge_density_ratio = np.empty((0,))
    non_rich_club_edge_density_ratio = np.empty((0,))
    local_edge_density_ratio = np.empty((0,))

    rich_club_edge_cost_ratio = np.empty((0,))
    non_rich_club_edge_cost_ratio = np.empty((0,))
    local_edge_cost_ratio = np.empty((0,))

    rich_club_edge_flow_ratio = np.empty((0,))
    non_rich_club_edge_flow_ratio = np.empty((0,))
    local_edge_flow_ratio = np.empty((0,))

    rich_club_edge_time_ratio = np.empty((0,))
    non_rich_club_edge_time_ratio = np.empty((0,))
    local_edge_time_ratio = np.empty((0,))

    rich_club_count = np.empty((0,))
    non_rich_club_count = np.empty((0,))
    local_count = np.empty((0,))

    in_degree = G.adj.sum(axis=0)
    out_degree = G.adj.sum(axis=1)
    
    for k in range(max(in_degree.max(), out_degree.max())):

        rich_club_edges = np.outer(np.where(out_degree > k, 1, 0),
                                   np.where(in_degree > k, 1, 0)) * G.adj

        local_edges = np.outer(np.where(out_degree < k, 1, 0),
                               np.where(in_degree < k, 1, 0)) * G.adj

        non_rich_club_edges = (1 - np.outer(np.where(out_degree > k, 1, 0),
                                            np.where(in_degree > k, 1, 0))) * G.adj

        rich_club_count = np.append(rich_club_count, rich_club_edges.sum())
        non_rich_club_count = np.append(non_rich_club_count, non_rich_club_edges.sum())
        local_count = np.append(local_count, local_edges.sum())

        rich_club_edge_density_ratio = np.append(rich_club_edge_density_ratio,
                                                 G.adj_weights[rich_club_edges == 1].sum() /
                                                 G.adj_weights.sum() / rich_club_edges.sum())

        non_rich_club_edge_density_ratio = np.append(non_rich_club_edge_density_ratio,
                                                     G.adj_weights[non_rich_club_edges == 1].sum() /
                                                     G.adj_weights.sum() / non_rich_club_edges.sum())

        local_edge_density_ratio = np.append(local_edge_density_ratio,
                                             G.adj_weights[local_edges == 1].sum() /
                                             G.adj_weights.sum() / local_edges.sum())

        rich_club_edge_cost_ratio = np.append(rich_club_edge_cost_ratio,
                                              G.wiring_cost[rich_club_edges == 1].sum() /
                                              G.wiring_cost.sum() / rich_club_edges.sum())

        non_rich_club_edge_cost_ratio = np.append(non_rich_club_edge_cost_ratio,
                                                  G.wiring_cost[non_rich_club_edges == 1].sum() /
                                                  G.wiring_cost.sum() / non_rich_club_edges.sum())

        local_edge_cost_ratio = np.append(local_edge_cost_ratio,
                                          G.wiring_cost[local_edges == 1].sum() /
                                          G.wiring_cost.sum() / local_edges.sum())

        rich_club_edge_flow_ratio = np.append(rich_club_edge_flow_ratio,
                                              G.WEflowsLinear_edge[rich_club_edges == 1].sum() /
                                              G.total_flow / rich_club_edges.sum())

        non_rich_club_edge_flow_ratio = np.append(non_rich_club_edge_flow_ratio,
                                                  G.WEflowsLinear_edge[non_rich_club_edges == 1].sum() /
                                                  G.total_flow / non_rich_club_edges.sum())

        local_edge_flow_ratio = np.append(local_edge_flow_ratio,
                                          G.WEflowsLinear_edge[local_edges == 1].sum() /
                                          G.total_flow / local_edges.sum())

        rich_club_edge_time_ratio = np.append(rich_club_edge_time_ratio,
                                              G.WEcostsLinear_ratio[rich_club_edges == 1].sum() /
                                              G.total_flow / rich_club_edges.sum())

        non_rich_club_edge_time_ratio = np.append(non_rich_club_edge_time_ratio,
                                                  G.WEcostsLinear_ratio[non_rich_club_edges == 1].sum() /
                                                  G.total_flow / non_rich_club_edges.sum())

        local_edge_time_ratio = np.append(local_edge_time_ratio,
                                          G.WEcostsLinear_ratio[local_edges == 1].sum() /
                                          G.total_flow / local_edges.sum())

    df_edge = pd.DataFrame(data={'rich_club_count': rich_club_count,
                                     'non_rich_club_count': non_rich_club_count,
                                     'local_count': local_count,
                                     'rich_club_edge_density_ratio': rich_club_edge_density_ratio,
                                     'non_rich_club_edge_density_ratio': non_rich_club_edge_density_ratio,
                                     'local_edge_density_ratio': local_edge_density_ratio,
                                     'rich_club_edge_cost_ratio': rich_club_edge_cost_ratio,
                                     'non_rich_club_edge_cost_ratio': non_rich_club_edge_cost_ratio,
                                     'local_edge_cost_ratio': local_edge_cost_ratio,
                                     'rich_club_edge_flow_ratio': rich_club_edge_flow_ratio,
                                     'non_rich_club_edge_flow_ratio': non_rich_club_edge_flow_ratio,
                                     'local_edge_flow_ratio': local_edge_flow_ratio,
                                     'rich_club_edge_time_ratio': rich_club_edge_time_ratio,
                                     'non_rich_club_edge_time_ratio': non_rich_club_edge_time_ratio,
                                     'local_edge_time_ratio': local_edge_time_ratio},
                               index=range(max(in_degree.max(), out_degree.max())))

    return df_edge


if __name__ == "__main__":

    model = sys.argv[1]
    # load raw data
    G = load_g(model)

    # check if adj is symmetric
    if np.allclose(G.adj_weights, G.adj_weights.T):
        df_nodal = nodal_und(G)
        df_edge = edge_und(G)
    else:
        df_nodal = nodal_dir(G)
        df_edge = edge_dir(G)

    # save file
    with open('analysis/' + model + '_dfs.pickle', 'wb') as f:
        pickle.dump({'df_nodal':df_nodal, 'df_edge':df_edge}, f, protocol=pickle.HIGHEST_PROTOCOL)


