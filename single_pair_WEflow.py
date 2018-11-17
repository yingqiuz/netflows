import sys
import numpy as np
import pickle
from netflows import Graph
from netflows.funcs import WElinearsolve, WEaffinesolve, WEbprsolve



def construct_data():
    parameter_dict = {}
    # python single_pair_flow.py G=filename source=s target=t
    for user_input in sys.argv[1:]:
        varname = user_input.split("=")[0]
        varvalue = user_input.split("=")[1]
        parameter_dict[varname] = varvalue
    # load adj data
    with open('data/' + parameter_dict['G']+'_adj.pickle', 'rb') as f:
        adj = pickle.load(f)
    # load dist data
    try:
        with open('data/' + parameter_dict['G']+'_dist.pickle', 'rb') as f:
            dist = pickle.load(f)
    except:
        dist = np.zeros(adj.shape)

    return adj, dist, parameter_dict

if __name__ == '__main__':

    adj, dist, parameter_dict = construct_data()
    G = Graph(adj = adj, dist = dist, weights = adj)

    s = parameter_dict['source']
    t = parameter_dict['target']

    x, allflows, total_cost_sum, total_cost = WElinearsolve(G, int(s), int(t), cutoff=None, maximum_iter=100000, tol=1e-8)

    filename = 'results/'+ parameter_dict['G']+ '_WE_' + s + '_' + t + '.pickle'

    with open(filename, 'wb') as f:
        pickle.dump({'allflows':allflows, 'total_cost_sum':total_cost_sum, 'total_cost':total_cost},
                    f, protocol=pickle.HIGHEST_PROTOCOL)
