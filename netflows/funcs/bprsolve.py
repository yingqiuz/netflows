from __future__ import absolute_import
from netflows.funcs.costfuncs import BPR_cost, BPR_WE_obj, BPR_SO_obj

import numpy as np
import scipy

def WEbprsolve(G, s, t, tol = 1e-8, maximum_iter = 10000, cutoff = None, a = None, u = None):
    """
    single pair Wardrop Equilibrium flow
    s: source
    t: destination
    tol: tolerance
    gamma: descent speed
    """

    allpaths = G.allpaths[s][t]
    if allpaths == []:
        if cutoff == None:
            print("Warning: cutoff not specified. it may take hugh memory to find all paths")
            cutoff = min(G.adj.shape)
        allpaths = G.findallpaths(s, t, cutoff)

    if a is None:
        a = G.adj_dist

    if u is None:
        u = G.rpl_weights

    return _WEbprsolve(G, s, t, tol, maximum_iter, allpaths, a, u)

def SObprsolve(G, s, t, tol=1e-8, maximum_iter = 10000, cutoff = None, a = None, u = None):
    """
    :param G:
    :param s:
    :param t:
    :param tol:
    :param maximum_iter:
    :param cutoff:
    :return:
    """

    allpaths = G.allpaths[s][t]
    if allpaths == []:
        if cutoff == None:
            print("Warning: cutoff not specified. it may take hugh memory to find all paths")
            cutoff = min(G.adj.shape)
        allpaths = G.findallpaths(s, t, cutoff)

    if a is None:
        a = G.adj_dist

    if u is None:
        u = G.rpl_weights

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
    
    #weights = np.copy(u)
    #weights[G.adj_weights == 0] = np.inf
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

    #obj_fun = np.sum( BPR_WE_obj(allflows, a, weights))
    # obj_fun = np.sum( G.WE_obj(allflows), axis = None)
    total_cost = np.sum(allflows *  BPR_cost(allflows, a, u))
    #total_traveltime = np.sum( BPR_cost(allflows,  G.adj_dist, weights))
    print('The initial cost is %f, and the initial flow is ' % (total_cost), x)
    print('------solve the Wardrop Equilibrium------')
    gradients = np.array(
        [np.sum( BPR_cost(allflows, a, u) * (
                    path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1, 0) *
                    path_arrays[-1]))
         for k in range(num_variables - 1)]
    )
    
    # initial gamma determination
    gamma1 = np.min(np.abs( x[:-1] / gradients ))
    gamma2 = np.min(np.abs( (1 - x[:-1]) / gradients) )
    gamma = min(gamma1, gamma2) * 2 / 3

    for k in range(maximum_iter):  # maximal iteration 10000

        #prev_obj_fun = np.copy(obj_fun)
        prev_x = np.copy(x)
        #prev_allflows = np.copy(allflows)
        #prev_total_cost = np.copy(total_cost)
        #prev_total_traveltime = np.copy(total_traveltime)
        prev_gradients = np.copy(gradients)

        # update x
        #print(gradients)
        x[:-1] = prev_x[:-1] - gamma * gradients
        x[-1] = 1 - np.sum(x[:-1])  # the flow in the last path

        if np.sum(np.where(x < 0, 1, 0)) > 0:  # flow in at least one path is negtive
            gradients[x[:-1] < 0] = prev_x[:-1][x[:-1] < 0] / gamma
            # neg = x[x < 0].sum()
            # gamma1 = np.min(np.abs(prev_x[:-1] / gradients))
            # gamma2 = np.min(np.abs((1 - prev_x[:-1]) / gradients))
            # gamma3 = gamma + np.abs(x[-1])/np.sum(gradients)
            # gamma = min(gamma1, gamma2)
            # print(gamma)
            x[:-1] = prev_x[:-1] - gamma * gradients
            x[-1] = 1 - np.sum(x[:-1])  # the flow in the last path
            # print('new x is ', x)
            if x[-1] < 0:
                # define new gradients, increase negative ones
                # how much increase to make sure they are within the constraints?
                # reduce the amount proportional to the original
                gradients[gradients < 0] += (np.abs(x[-1]) / gamma) * (
                        gradients[gradients < 0] / gradients[gradients < 0].sum())
                # gamma = gamma + np.abs(x[-1]) / np.sum(gradients)
                x[:-1] = prev_x[:-1] - gamma * gradients
                x[-1] = 1 - np.sum(x[:-1])  # the flow in the last path

        allflows = np.sum(path_arrays * x.reshape(num_variables, 1, 1), axis=0)
        # obj_fun = np.sum( BPR_WE_obj(allflows, a, u), axis=None)
        #diff_value = obj_fun - prev_obj_fun
        total_cost = np.sum(allflows *  BPR_cost(allflows, a, u), axis=None)

        print('Iteration %d: The total cost is %f, and the flow is ' % (k, total_cost), x)
        # new gradients
        gradients = np.array(
            [np.sum( BPR_cost(allflows, a, u) * (
                        path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1, 0) *
                        path_arrays[-1]))
             for k in range(num_variables - 1)]
        )
        # new gamma

        print(gradients-prev_gradients)
        if np.sum(np.where(np.abs(gradients-prev_gradients) < tol, 0, 1)) == 0:
            G.WEflowsBPR[s][t] = x
            G.WEcostsBPR[s][t] = total_cost
            G.WEflowsBPR_edge[s][t] = allflows
            return total_cost, x

        gamma = np.inner(x[:-1] - prev_x[:-1], gradients - prev_gradients) / \
                np.inner(gradients - prev_gradients, gradients - prev_gradients)

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
    #weights = np.copy(u)
    #weights[  G.adj_weights == 0 ] = np.inf
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

    obj_fun = np.sum( BPR_SO_obj(allflows, a, u))
    # total_cost = np.sum(allflows *  linear_cost(allflows, G.adj_dist))
    #total_traveltime = np.sum( BPR_cost(allflows, G.adj_dist, weights))
    print('The initial cost is %f, and the initial flow is ' % (obj_fun), x)
    print('------solve the system optimal flow------')
    gradients = np.array(
        [np.sum( BPR_cost(allflows, a, u) * (
                    path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1, 0) *
                    path_arrays[-1]))
         for k in range(num_variables - 1)]
    ) + np.array(
        [np.sum(allflows * a * (0.6 * (allflows * u) ** 3 * u) * (
                    path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1, 0) *
                    path_arrays[-1]))
         for k in range(num_variables - 1)]
    )

    # initial step size
    gamma1 = np.min(np.abs(x[:-1] / gradients))
    gamma2 = np.min(np.abs((1 - x[:-1]) / gradients))
    gamma = min(gamma1, gamma2) * 2 / 3

    for k in range(maximum_iter):  # maximal iteration 10000
        ######## TBC
        # prev_obj_fun = np.copy(obj_fun)
        # prev_x = np.copy(x)
        # prev_allflows = np.copy(allflows)
        # prev_total_traveltime = np.copy(total_traveltime)
        # gradients = np.array(
        #    [np.sum(  BPR_cost(prev_allflows, G.adj_dist, weights) * (path_arrays[k] * np.where(path_arrays[-1]==0, 1, 0) - np.where(path_arrays[k]==0, 1, 0) * path_arrays[-1]) ) 
        #     for k in range(num_variables - 1)]
        # ) + np.array(
        # [np.sum(allflows * G.adj_dist * (0.6 * ( allflows / weights ) ** 3 / weights ) * (path_arrays[k] * np.where(path_arrays[-1]==0, 1, 0) - np.where(path_arrays[k]==0, 1, 0) * path_arrays[-1])) 
        # for k in range(num_variables - 1)]
        # )
        # print(gradients)
        #prev_obj_fun = np.copy(obj_fun)
        prev_x = np.copy(x)
        #prev_allflows = np.copy(allflows)
        #prev_total_traveltime = np.copy(total_traveltime)
        prev_gradients = np.copy(gradients)

        # update x
        x[:-1] = prev_x[:-1] - gamma * gradients
        x[-1] = 1 - np.sum(x[:-1])  # the flow in the last path

        if np.sum(np.where(x < 0, 1, 0)) > 0:  # flow in at least one path is negtive
            gradients[x[:-1] < 0] = prev_x[:-1][x[:-1] < 0] / gamma
            # neg = x[x < 0].sum()
            # gamma1 = np.min(np.abs(prev_x[:-1] / gradients))
            # gamma2 = np.min(np.abs((1 - prev_x[:-1]) / gradients))
            # gamma3 = gamma + np.abs(x[-1])/np.sum(gradients)
            # gamma = min(gamma1, gamma2)
            # print(gamma)
            x[:-1] = prev_x[:-1] - gamma * gradients
            x[-1] = 1 - np.sum(x[:-1])  # the flow in the last path
            # print('new x is ', x)
            if x[-1] < 0:
                # define new gradients, increase negative ones
                # how much increase to make sure they are within the constraints?
                # reduce the amount proportional to the original
                gradients[gradients < 0] += (np.abs(x[-1]) / gamma) * (
                        gradients[gradients < 0] / gradients[gradients < 0].sum())
                # gamma = gamma + np.abs(x[-1]) / np.sum(gradients)
                x[:-1] = prev_x[:-1] - gamma * gradients
                x[-1] = 1 - np.sum(x[:-1])  # the flow in the last path

        allflows = np.sum(path_arrays * x.reshape(num_variables, 1, 1), axis=0)
        obj_fun = np.sum( BPR_SO_obj(allflows, a, u), axis=None)
        #diff_value = obj_fun - prev_obj_fun
        #total_traveltime = np.sum( BPR_cost(allflows, G.adj_dist, weights), axis=None)
        print('Iteration %d: The total cost is %f, and the flow is ' % (k, obj_fun), x)
        # update gradients
        #prev_gradients = np.copy(gradients)
        gradients = np.array(
            [np.sum( BPR_cost(allflows, a, u) * (
                        path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1, 0) *
                        path_arrays[-1]))
             for k in range(num_variables - 1)]
        ) + np.array(
            [np.sum(allflows * a * (0.6 * (allflows * u) ** 3 * u) * (
                        path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1, 0) *
                        path_arrays[-1]))
             for k in range(num_variables - 1)]
        )

        # convergence?
        if np.sum(np.where(np.abs(gradients-prev_gradients) < tol, 0, 1)) == 0:
            G.SOflowsBPR[s][t] = x
            G.SOcostsBPR[s][t] = obj_fun
            G.SOflowsBPR_edge[s][t] = allflows
            return obj_fun, x
        # new step size
        gamma = np.inner(x[:-1] - prev_x[:-1], gradients - prev_gradients) / \
                np.inner(gradients - prev_gradients, gradients - prev_gradients)

    print('global minimum not found')
    return