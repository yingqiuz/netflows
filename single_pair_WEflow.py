import sys
import numpy as np
import pickle
from netflows import Graph
from netflows.funcs import WElinearsolve, WEaffinesolve, WEbprsolve


def construct_data():
    parameters = {}
    # python single_pair_flow.py adj=filename dist=filename source=s target=t
    for user_input in sys.argv[1:]:
        varname = user_input.split("=")[0]
        varvalue = user_input.split("=")[1]
        parameters[varname] = varvalue
    # load adj data
    with open('data/' + parameters['adj'] + '_adj.pickle', 'rb') as f:
        adj = pickle.load(f)
    # load dist data
    try:
        with open('data/' + parameters['dist'] + '_dist.pickle', 'rb') as f:
            dist = pickle.load(f)
    except:
        dist = np.zeros(adj.shape)

    return adj, dist, parameters


if __name__ == '__main__':
    adj, dist, parameters = construct_data()
    G = Graph(adj=adj, dist=dist, weights=adj)

    s = parameters['s']
    t = parameters['t']
    try:
        x, allflows, total_cost_sum, total_cost = WElinearsolve(G, int(s), int(t), cutoff=None, maximum_iter=100000,
                                                            tol=1e-8)

        filename = 'results/' + parameters['adj'] + '_WE_' + s + '_' + t + '.pickle'

        with open(filename, 'wb') as f:
            pickle.dump({'allflows': allflows, 'total_cost_sum': total_cost_sum, 'total_cost': total_cost},
                        f, protocol=pickle.HIGHEST_PROTOCOL)

    except:
        print('no WE found')
