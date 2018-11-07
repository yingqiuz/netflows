from __future__ import absolute_import
from netflows.funcs.costfuncs import linear_cost, linear_WE_obj, linear_SO_obj

import numpy as np

def WElinearsolve(G, s, t, tol = 1e-8, maximum_iter = 10000, cutoff = None, a = None):
    """
    single pair Wardrop Equilibrium flow
    s: source
    t: destination
    tol: tolerance
    gamma: descent speed
    """
    allpaths = G.allpaths[s][t]
    if allpaths == []:
        if cutoff is None:
            print("Warning: cutoff not specified. it may take hugh memory to find all paths")
            cutoff = min(G.adj.shape)

        allpaths = G.findallpaths(s, t, cutoff)

    if a is None:
        a = G.dist_weight_ratio

    return _WElinearsolve(G, s, t, tol, maximum_iter, allpaths, a)

def SOlinearsolve(G, s, t, tol=1e-8, maximum_iter = 10000, cutoff = None, a = None):
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
        if cutoff is None:
            print("Warning: cutoff not specified. it may take hugh memory to find all paths")
            cutoff = min(G.adj.shape)

        allpaths = G.findallpaths(s, t, cutoff)

    if a is None:
        a = G.dist_weight_ratio

    return _SOlinearsolve(G, s, t, tol, maximum_iter, allpaths, a)

def _WElinearsolve(G, s, t, tol, maximum_iter, allpaths, a):

    num_variables = len(allpaths) # the number of paths from s to t

    # x is the flow vector (path formulation)
    x = np.ones((num_variables, )) / num_variables # initial value
    # find equilibrium -- convex optimization
    # map paths to matrix
    path_arrays = np.empty((0, G.adj.shape[0], G.adj.shape[1])) # list of matrix to store path flows

    for path in allpaths:

        path_array_tmp = np.zeros(G.adj.shape)
        index_x = [path[k] for k in range(len(path)-1)] # x index of the adj matrix
        index_y = [path[k] for k in range(1, len(path))] # y index of the adj matrix
        path_array_tmp[index_x, index_y] = 1
        path_arrays = np.append(path_arrays, path_array_tmp[np.newaxis, :], axis = 0)

    # element (i, j) is the total flow on edge (i,j)
    allflows = np.sum(path_arrays * x.reshape(num_variables, 1, 1), axis = 0)

    #obj_fun = np.sum( linear_WE_obj(allflows, a))
    # obj_fun = np.sum(G.WE_obj(allflows), axis = None)
    total_cost = np.sum(allflows *  linear_cost(allflows, a))
    #total_traveltime = np.sum( linear_cost(allflows, a))
    print('The initial cost is %f, and the initial flow is ' % (total_cost), x)
    print('------solve the Wardrop Equilibrium------')

    gradients = np.array(
        [np.sum( linear_cost(allflows, a) * (
                    path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1, 0) *
                    path_arrays[-1]))
         for k in range(num_variables - 1)]
    )
    # print('the gradients are')
    # print(gradients)
    # print('the gamma is')
    # print(gamma)

    # initial step size determination
    gamma1 = np.min(np.abs(x[:-1] / gradients))
    gamma2 = np.min(np.abs((1 - x[:-1]) / gradients))
    gamma = min(gamma1, gamma2) * 2 / 3
    #print(min(gamma1, gamma2) )
    #gamma = 1e-8

    for k in range(maximum_iter):  # maximal iteration 10000

        #prev_obj_fun = np.copy(obj_fun)
        prev_x = np.copy(x)
        #prev_allflows = np.copy(allflows)

        #prev_total_cost = np.copy(total_cost)
        #prev_total_traveltime = np.copy(total_traveltime)
        prev_gradients = np.copy(gradients)

        # update x
        x[:-1] = prev_x[:-1] - gamma * gradients
        x[-1] = 1 - np.sum(x[:-1])  # the flow in the last path

        if np.sum(np.where(x < 0, 1, 0)) > 0:  # flow in at least one path is negtive
            #print('One of the flows reaches zero')
            #print('Iteration %d: The total cost is %f, and the flow is ' % (k, total_cost), prev_x)
            # store flow and cost
            #G.WEflowsLinear[s][t] = prev_x
            #G.WEcostsLinear[s][t] = total_cost
            # store flow in edge formulation
            #G.WEflowsLinear_edge[s][t] = allflows
            #return total_cost, prev_x
            gamma1 = np.min(np.abs(x[:-1] / gradients))
            gamma2 = np.min(np.abs((1 - x[:-1]) / gradients))
            gamma = min(gamma1, gamma2) * 2 / 3
            x[:-1] = prev_x[:-1] - gamma * gradients
            x[-1] = 1 - np.sum(x[:-1])  # the flow in the last path

        # update allflows, obj func
        allflows = np.sum(path_arrays * x.reshape(num_variables, 1, 1), axis=0)
        #obj_fun = np.sum( linear_WE_obj(allflows, a), axis=None)
        #diff_value = obj_fun - prev_obj_fun
        total_cost = np.sum(allflows *  linear_cost(allflows, a), axis=None)
        #total_traveltime = np.sum( linear_cost(allflows, a), axis=None)

        # new gradients and learning rate
        gradients = np.array(
            [np.sum( linear_cost(allflows, a) * (
                        path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1, 0) *
                        path_arrays[-1]))
             for k in range(num_variables - 1)]
        )

        # print('the gradients are')
        # print(gradients)
        # print('the gamma is')
        # print(gamma)
        if np.sum(np.where(np.abs(gradients) < tol, 0, 1)) == 0:
            G.WEflowsLinear[s][t] = x
            G.WEcostsLinear[s][t] = total_cost
            G.WEflowsLinear_edge[s][t] = allflows
            print('Iteration %d: The total cost is %f, and the flow is ' % (k, total_cost), x)
            return total_cost, x

        gamma = np.inner(x[:-1] - prev_x[:-1], gradients - prev_gradients) / \
                np.inner(gradients - prev_gradients, gradients - prev_gradients)

    print('global minimum not found')
    return

def _SOlinearsolve(G, s, t, tol, maximum_iter, allpaths, a):
    """
    single pair System Optimal flow
    s: source
    t: destination
    tol: tolerance
    gamma: descent speed
    """
    num_variables = len(allpaths)  # the number of paths from s to t

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

    obj_fun = np.sum( linear_SO_obj(allflows, a))
    # total_cost = np.sum(allflows *  linear_cost(allflows, G.adj_dist))
    #total_traveltime = np.sum( linear_cost(allflows, G.dist_weight_ratio))
    print('The initial cost is %f, and the initial flow is ' % (obj_fun), x)
    print('------solve the system optimal flow------')

    gradients = np.array(
        [np.sum( linear_cost(allflows, a) * (
                    path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1, 0) *
                    path_arrays[-1]))
         for k in range(num_variables - 1)]
    ) + np.array(
        [np.sum(allflows * a * (
                    path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1, 0) *
                    path_arrays[-1]))
         for k in range(num_variables - 1)]
    )

    # initial step size determination
    gamma1 = np.min( np.abs( x[:-1]/gradients ) )
    gamma2 = np.min( np.abs( (1 - x[:-1])/ gradients ) )
    gamma = min(gamma1, gamma2) * 2  / 3

    for k in range(maximum_iter):  # maximal iteration 10000

        #prev_obj_fun = np.copy(obj_fun)
        prev_x = np.copy(x)
        #prev_allflows = np.copy(allflows)
        #prev_total_traveltime = np.copy(total_traveltime)
        prev_gradients = np.copy(gradients)

        # update x
        x[:-1] = prev_x[:-1] - gamma * gradients
        x[-1] = 1 - np.sum(x[:-1])  # the flow in the last path

        if np.sum(np.where(x < 0, 1, 0)) > 0:  # flow in at least one path is negtive
            #print('One of the flows reaches zero')
            #print('Iteration %d: The total cost is %f, and the flow is ' % (k, obj_fun), prev_x)
            #G.SOflowsLinear[s][t] = prev_x
            #G.SOcostsLinear[s][t] = obj_fun
            #G.SOflowsLinear_edge[s][t] = allflows
            #return obj_fun, prev_x
            gamma1 = np.min(np.abs(x[:-1] / gradients))
            gamma2 = np.min(np.abs((1 - x[:-1]) / gradients))
            gamma = min(gamma1, gamma2) * 2 / 3
            x[:-1] = prev_x[:-1] - gamma * gradients
            x[-1] = 1 - np.sum(x[:-1])  # the flow in the last path

        allflows = np.sum(path_arrays * x.reshape(num_variables, 1, 1), axis=0)
        obj_fun = np.sum( linear_SO_obj(allflows, a), axis=None)
        #diff_value = obj_fun - prev_obj_fun
        #total_traveltime = np.sum( linear_cost(allflows, G.dist_weight_ratio), axis=None)

        # new gradients and gamma 
        gradients = np.array(
            [np.sum( linear_cost(allflows, a) * (
                        path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1, 0) *
                        path_arrays[-1]))
             for k in range(num_variables - 1)]
        ) + np.array(
            [np.sum(allflows * a * (
                        path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1, 0) *
                        path_arrays[-1]))
             for k in range(num_variables - 1)]
        )

        if np.sum(np.where(np.abs(gradients) < tol, 0, 1)) == 0:
            G.SOflowsLinear[s][t] = x
            G.SOcostsLinear[s][t] = obj_fun
            G.SOflowsLinear_edge[s][t] = allflows
            print('Iteration %d: The total cost is %f, and the flow is ' % (k, obj_fun), x)
            return obj_fun, x

        gamma = np.inner(x[:-1] - prev_x[:-1], gradients - prev_gradients) \
                / np.inner(gradients - prev_gradients, gradients - prev_gradients)

    print('global minimum not found')
    return