#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys,os
import numpy as np
import pickle


def read_data(model):
    """
    :param model: such as 'mouse', 'mouse_lat'
    :return:
    """
    filedir = 'results/' + model
    filelist = [f for f in os.listdir(filedir) if os.path.isfile(os.path.join(filedir, f))]

    with open('data/' + model + '_adj.pickle', 'rb') as f:
        adj = pickle.load(f)

    return filelist, adj

if __name__ == '__main__':

    # read input
    model = sys.argv[1]
    filelist, adj = read_data(model)

    # initialize total flow and cost
    total_flow = 0
    total_flow_edge = np.zeros(adj.shape)
    total_time = np.zeros(adj.shape)
    total_time_edge = np.zeros(adj.shape)
    total_time_ratio = np.zeros(adj.shape)

    for fname in filelist:
        print(fname)
        filename = 'results/' + model + '/' + fname
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        stpair = fname.split('_WE_')[1]
        s = stpair.split('_')[0]
        t = stpair.split('_')[1].split('.')[0]

        print(s, t)

        allflows = data['allflows']
        total_cost_sum = data['total_cost_sum']
        total_cost = data['total_cost']

        if total_cost_sum > 0:
            total_flow += 1
            total_flow_edge += allflows
            total_time[int(s), int(t)] = total_cost_sum
            total_time_edge += total_cost
            total_time_ratio += total_cost / total_cost_sum

    with open('total/' + model + '.pickle', 'wb') as f:
        pickle.dump({'total_flow':total_flow, 'total_flow_edge':total_flow_edge,
                     'total_time':total_time, 'total_time_edge':total_time_edge,
                     'total_time_ratio':total_time_ratio},
                    f, protocol=pickle.HIGHEST_PROTOCOL)






