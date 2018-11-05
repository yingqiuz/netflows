from __future__ import absolute_import
from netflows.funcs.costfuncs import BPR_cost, BPR_WE_obj, BPR_SO_obj

import numpy as np
import scipy

def WEbprsolve(G, s, t, tol = 1e-12, maximum_iter = 10000, cutoff = None, a = None, u = None):
    """
    single pair Wardrop Equilibrium flow
    s: source
    t: destination
    tol: tolerance
    gamma: descent speed
    """
    if cutoff == None:
        print("Warning: cutoff not specified. it may take hugh memory to find all paths")
        cutoff = min(G.adj.shape)

    allpaths = G.allpaths[s][t]
    if allpaths == []:
        allpaths = G.findallpaths(s, t, cutoff)

    if a == None:
        a = G.adj_dist
    # binarize dist matrix
    a[G.adj == 0] = 0

    if u == None:
        u = G.adj_weights

    return _WEbprsolve(G, s, t, tol, maximum_iter, allpaths, a, u)

def SObprsolve(G, s, t, tol=1e-12, maximum_iter = 10000, cutoff = None, a = None, u = None):
    """
    :param G:
    :param s:
    :param t:
    :param tol:
    :param maximum_iter:
    :param cutoff:
    :return:
    """
    if cutoff == None:
        print("Warning: cutoff not specified. it may take hugh memory to find all paths")
        cutoff = min(G.adj.shape)

    allpaths = G.allpaths[s][t]
    if allpaths == []:
        allpaths = G.findallpaths(s, t, cutoff)

    if a == None:
        a = G.adj_dist
    # binarize dist matrix
    a[G.adj == 0 ] = 0

    if u == None:
        u = G.adj_weights

    return _SObprsolve(G, s, t, tol, maximum_iter, allpaths, a, u)

def _WEbprsolve(G, s, t, tol, maximum_iter, allpaths, a, u):
    """
    single pair Wardrop Equilibrium flow, BPR cost function
    s: source
    t: destination
    tol: tolerance
    gamma: descent speed
    """

    num_variables = len(allpaths)  # the number of paths from s to t
    
    weights = np.copy(u)
    weights[G.adj == 0] = np.inf
    # x is the flow vector (path formulation)
    x = np.ones((num_variables,)) / num_variables  # initial value
    # find equilibrium -- convex optimization
    # map to matrix
    path_arrays = np.empty((0, G.adj.shape[0], G.adj.shape[1]))  # list of matrix to store path flows

    for path in allpaths:
        path_array_tmp = np.zeros(G.adj.shape)
        index_x = [path[k] for k in range(len(path) - 1)]  # x index of the adj matrix
        index_y = [path[k] for k in range(1, len(path))]  # y index of the adj matrix
        path_array_tmp[index_x, index_y] = 1
        path_arrays = np.append(path_arrays, path_array_tmp[np.newaxis, :], axis=0)

    # element (i, j) is the total flow on edge (i,j)    
    allflows = np.sum(path_arrays * x.reshape(num_variables, 1, 1), axis=0)
    allflows[G.adj == 0] = 0 #seems unnecessary... ?

    obj_fun = np.sum(BPR_WE_obj(allflows, a, weights))
    # obj_fun = np.sum(G.WE_obj(allflows), axis = None)
    total_cost = np.sum(allflows * BPR_cost(allflows, a, weights))
    #total_traveltime = np.sum( BPR_cost(allflows, G.adj_dist, weights))
    print('initial cost is %f' % total_cost)
    print('initial flows are', x)
    print('------solve the Wardrop Equilibrium------')
    
    gradients = np.array(
        [np.sum(BPR_cost(allflows, a, weights) * (
                    path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1, 0) *
                    path_arrays[-1]))
         for k in range(num_variables - 1)]
    )

    for k in range(maximum_iter):  # maximal iteration 10000

        prev_obj_fun = np.copy(obj_fun)
        prev_x = np.copy(x)
        #prev_allflows = np.copy(allflows)
        #prev_total_cost = np.copy(total_cost)
        #prev_total_traveltime = np.copy(total_traveltime)
        #prev_gradients = np.copy(gradients)

        # update x
        #print(gradients)
        result = scipy.optimize.linprog(gradients, bounds=(0, 1),
                                        options={'maxiter': 1000, 'disp': False, 'tol': 1e-12, 'bland': True})

        # step size determination
        gamma = 2 / (k + 1 + 2)
        # update x
        x[:-1] = prev_x[:-1] + gamma * (result.x - x[:-1])
        x[-1] = 1 - np.sum(x[:-1])  # the flow in the last path

        #if np.sum() < tol:  # flow in at least one path is negtive
        #    print('Iteration %d: The total cost is %f, the total travel time is %f, and the flow is ' % (
        #    k, prev_total_cost, prev_total_traveltime), prev_x)
        #    G.WEflowsBPR[s][t] = prev_x
        #    G.WEcostsBPR[s][t] = prev_total_cost
        #    G.WEflowsBPR_edge[s][t] = prev_allflows
        #    return prev_total_cost, prev_total_traveltime, prev_x

        allflows = np.sum(path_arrays * x.reshape(num_variables, 1, 1), axis=0)
        allflows[G.adj == 0] = 0
        obj_fun = np.sum( BPR_WE_obj(allflows, a, weights), axis = None)
        diff_value = obj_fun - prev_obj_fun
        diff_value_x = x - prev_x
        total_cost = np.sum(allflows *  BPR_cost(allflows, a, weights), axis=None)

        if np.abs(diff_value) < np.abs(prev_obj_fun * tol) and np.abs(diff_value_x) < np.abs( tol * prev_x):
            print('Wardrop equilibrium found. total cost %f' % total_cost)
            print('the flows are', x)
            G.WEflowsBPR[s][t] = x
            G.WEcostsBPR[s][t] = total_cost
            G.WEflowsBPR_edge[s][t] = allflows
            return total_cost, x
        #total_traveltime = np.sum( BPR_cost(allflows, G.adj_dist, weights), axis=None)
        #print('Iteration %d: The total cost is %f, the total travel time is %f, and the flow is ' % (
        # k, total_cost, total_traveltime), x)

        # new gradients
        gradients = np.array(
            [np.sum( BPR_cost(allflows, a, weights) * (
                        path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1,
                                                                                         0) * path_arrays[-1]))
             for k in range(num_variables - 1)]
        )
        # new gamma
        #gamma = np.inner(x[:-1] - prev_x[:-1], gradients - prev_gradients) / np.inner(gradients - prev_gradients,
                                                                                      #gradients - prev_gradients)

    print('global minimum not found')
    return


def _SObprsolve(G, s, t, tol, maximum_iter, allpaths, a, u):
    """
    :param G:
    :param s:
    :param t:
    :param tol:
    :param maximum_iter:
    :param allpaths:
    :param u:
    :return:
    """
    num_variables = len(allpaths)  # the number of paths from s to t
    weights = np.copy(u)
    weights[  G.adj_weights == 0 ] = np.inf
    # x is the flow vector (path formulation)
    x = np.ones((num_variables,)) / num_variables  # initial value
    # find equilibrium -- convex optimization
    # map to matrix
    path_arrays = np.empty((0,   G.adj.shape[0],   G.adj.shape[1]))  # list of matrix to store path flows

    for path in allpaths:
        path_array_tmp = np.zeros(  G.adj.shape)
        index_x = [path[k] for k in range(len(path) - 1)]  # x index of the adj matrix
        index_y = [path[k] for k in range(1, len(path))]  # y index of the adj matrix
        path_array_tmp[index_x, index_y] = 1
        path_arrays = np.append(path_arrays, path_array_tmp[np.newaxis, :], axis=0)

    # element (i, j) is the total flow on edge (i,j)    
    allflows = np.sum(path_arrays * x.reshape(num_variables, 1, 1), axis=0)
    allflows[G.adj == 0] = 0

    obj_fun = np.sum(  BPR_SO_obj(allflows, a, weights) )
    # total_cost = np.sum(allflows *   linear_cost(allflows,   G.adj_dist))
    #total_traveltime = np.sum(  BPR_cost(allflows,  a, weights))
    #print(
    #'The initial cost is %f, the initial travel time is %f, and the initial flow is ' % (obj_fun, total_traveltime), x)
    print('initial cost is %f' % obj_fun)
    print('initial flows are', x)
    print('------solve the system optimal flow------')

    gradients = np.array(
        [np.sum(  BPR_cost(allflows, a, weights) * (
                    path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1, 0) *
                    path_arrays[-1]))
         for k in range(num_variables - 1)]
    ) + np.array(
        [np.sum(allflows * a * (0.6 * (allflows / weights) ** 3 / weights) * (
                    path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1, 0) *
                    path_arrays[-1]))
         for k in range(num_variables - 1)]
    )

    for k in range(maximum_iter):  # maximal iteration 10000
        ######### TBC

        prev_obj_fun = np.copy(obj_fun)
        prev_x = np.copy(x)
        #prev_allflows = np.copy(allflows)
        #prev_total_traveltime = np.copy(total_traveltime)

        # update x
        result = scipy.optimize.linprog(gradients, bounds=(0, 1),
                                        options={'maxiter': 1000, 'disp': False, 'tol': 1e-12, 'bland': True})

        # step size determination
        gamma = 2 / (k + 1 + 2)
        # update x
        x[:-1] = prev_x[:-1] + gamma * (result.x - x[:-1])
        x[-1] = 1 - np.sum(x[:-1])  # the flow in the last path

        # update
        allflows = np.sum(path_arrays * x.reshape(num_variables, 1, 1), axis=0)
        allflows[G.adj == 0] = 0
        obj_fun = np.sum(  BPR_SO_obj(allflows, a, weights), axis=None)
        diff_value = obj_fun - prev_obj_fun
        diff_value_x = x - prev_x

        # convergence
        if np.abs(diff_value) < np.abs(prev_obj_fun * tol) and np.abs(diff_value_x) < np.abs( tol * prev_x):
            print('system optimum found: total cost %f' % obj_fun)
            print('the flows are', x)
            G.SOflowsBPR[s][t] = x
            G.SOcostsBPR[s][t] = obj_fun
            G.SOflowsBPR_edge[s][t] = allflows
            return obj_fun, x

        #prev_gradients = np.copy(gradients)
        gradients = np.array(
            [np.sum(  BPR_cost(allflows, a, weights) * (
                        path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1, 0) *
                        path_arrays[-1]))
             for k in range(num_variables - 1)]
        ) + np.array(
            [np.sum(allflows * a * (0.6 * (allflows / weights) ** 3 / weights) * (
                        path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1, 0) *
                        path_arrays[-1]))
             for k in range(num_variables - 1)]
        )
        # new learning rate
        # gamma = np.inner(x[:-1] - prev_x[:-1], gradients - prev_gradients) / np.inner(gradients - prev_gradients,
                                                                                     # gradients - prev_gradients)
        # convergence?

    print('global minimum not found')
    return