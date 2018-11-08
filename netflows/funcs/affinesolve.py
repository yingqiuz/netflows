from __future__ import absolute_import
from netflows.funcs.costfuncs import affine_cost,affine_SO_obj,affine_WE_obj, affine_we_obj_search, affine_so_obj_search
from netflows.funcs.gradfuncs import we_affine_grad,so_affine_grad

import numpy as np

def WEaffinesolve(G, s, t, tol = 1e-12, maximum_iter = 10000, cutoff = None, a = None, a0 = None):
    """
    single pair Wardrop Equilibrium flow
    s: source
    t: destination
    tol: tolerance
    gamma: descent speed
    """

    if cutoff is None:
        print("Warning: cutoff not specified. it may take hugh memory to find all paths")
        cutoff = min(G.adj.shape)
    # find all paths
    allpaths = G.findallpaths(s, t, cutoff)


    if a is None:
        a = G.rpl_weights

    if a0 is None:
        a0 = G.adj_dist

    return _WEaffinesolve(G, s, t, tol, maximum_iter, allpaths, a, a0)

def SOaffinesolve(G, s, t, tol=1e-12, maximum_iter = 10000, cutoff = None, a = None, a0 = None):
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
        # find all paths less than cutoff
        if cutoff == None:
            print("Warning: cutoff not specified. it may take hugh memory to find all paths")
            cutoff = min(G.adj.shape)
        allpaths = G.findallpaths(s, t, cutoff)

    if a is None:
        a = G.rpl_weights

    if a0 is None:
        a0 = G.adj_dist

    return _SOaffinesolve(G, s, t, tol, maximum_iter, allpaths, a, a0)

def _WEaffinesolve(G, s, t, tol, maximum_iter, allpaths, a, a0):
    """
    :param G:
    :param s:
    :param t:
    :param tol:
    :param maximum_iter:
    :param allpaths:
    :return:
    """
    #a0 = G.adj_weights

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

    # objective function to minimize
    obj_fun = affine_WE_obj(allflows, a, a0).sum()
    # total cost (i.e., flow * cost_func(flow) )
    total_cost = (allflows *  affine_cost(allflows, a, a0)).sum()
    #total_traveltime = np.sum( affine_cost(allflows, G.adj_dist, a0=a0)) useless for now...
    
    print('The initial cost is %f, and the initial flow is ' % (total_cost), x)
    print('------solve the Wardrop Equilibrium------')
    # initial gradients
    gradients = np.array(
        [np.sum(affine_cost(allflows, a, a0) * (
                    path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1, 0) *
                    path_arrays[-1]))
         for k in range(num_variables - 1)]
    )

    # initial estimation of step size gamma
    gamma1 = np.min( np.abs( x[:-1] / gradients ) )
    gamma2 = np.min( np.abs( (1 - x[:-1]) / gradients ) )
    gamma = min(gamma1, gamma2) * 2 / 3 # to make sure the flow on each path is positive after initial iteration

    for k in range(maximum_iter):  # maximal iteration 10000
        # store previous values
        prev_obj_fun = np.copy(obj_fun)
        prev_x = np.copy(x)
        prev_gradients = np.copy(gradients)
        
        # update x
        x[:-1] = prev_x[:-1] - gamma * gradients
        x[-1] = 1 - np.sum(x[:-1])  # the flow in the last path
        
        # if at least one of the flows is negative, re-adapt the gradients
        if np.sum(np.where(x < 0, 1, 0)) > 0:  
            # reduce positive gradients
            gradients[x[:-1] < 0] = prev_x[:-1][x[:-1] < 0] / gamma
            
            x[:-1] = prev_x[:-1] - gamma * gradients
            x[-1] = 1 - np.sum(x[:-1])  # the flow in the last path
            # if  the last one is still negative
            if x[-1] < 0:
                # define new gradients, increase negative ones
                # reduce the amount proportional to the original
                gradients[gradients < 0] += (np.abs(x[-1]) / gamma) * (
                            gradients[gradients < 0] / gradients[gradients < 0].sum())
                x[:-1] = prev_x[:-1] - gamma * gradients
                x[-1] = 1 - np.sum(x[:-1])  # the flow in the last path

        # update allflows and travel cost according to path formulation x
        allflows = np.sum(path_arrays * x.reshape(num_variables, 1, 1), axis=0)
        obj_fun = affine_WE_obj(allflows, a, a0).sum() #value of obj func. useless
        diff_value = obj_fun - prev_obj_fun #useless.....
        total_cost = (allflows * affine_cost(allflows, a, a0)).sum()
        
        print('Iteration %d: The total cost is %f, and the flow is ' % (k, total_cost), x)
        # new gradients and stepsize
        gradients = np.array(
            [np.sum( affine_cost(allflows, a, a0) * (
                        path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1, 0) *
                        path_arrays[-1]))
             for k in range(num_variables - 1)]
        )
        
        if np.where(np.abs(gradients - prev_gradients) < tol * np.abs(prev_gradients), 0, 1).sum() == 0: # convergence
            print('Wardrop equilibrium found:')
            print('Iteration %d: The total cost is %f, and the flow is ' % (k, total_cost), x)
            G.WEflowsAffine[s][t] = x
            G.WEcostsAffine[s][t] = total_cost
            G.WEflowsAffine_edge[s][t] = allflows
            return total_cost, x
        
        # new step size
        gamma = np.inner(x[:-1] - prev_x[:-1], gradients - prev_gradients) / \
                np.inner(gradients - prev_gradients, gradients - prev_gradients)

    print('global minimum not found')
    return

def _SOaffinesolve(G, s, t, tol, maximum_iter, allpaths, a, a0):
    """
    single pair System Optimal flow, affine cost function
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


    obj_fun = affine_SO_obj(allflows, a, a0).sum()  # objective function is the total cost function
  
    print('The initial cost is %f, and the initial flow is ' % (obj_fun), x)
    print('------solve the system optimal flow------')

    # initial gradients
    gradients = np.array(
        [np.sum(  affine_cost(allflows, a, a0) * (
                    path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1, 0) *
                    path_arrays[-1]))
         for k in range(num_variables - 1)]
    ) + np.array(
        [np.sum(allflows *  a * (
                    path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1, 0) *
                    path_arrays[-1]))
         for k in range(num_variables - 1)]
    )

    # initial step size determination
    gamma1 = np.min(np.abs(x[:-1] / gradients))
    gamma2 = np.min(np.abs((1 - x[:-1]) / gradients))
    #gamma3 = min(gamma1, gamma2) + np.abs(1 - np.sum(x[:-1]) + min(gamma1, gamma2)
    #                                      * np.sum(gradients)) / np.sum(gradients)
    gamma = min(gamma1, gamma2) * 2 / 3

    for k in range(maximum_iter):  # maximal iteration 10000

        prev_obj_fun = np.copy(obj_fun)
        prev_x = np.copy(x)
        prev_gradients = np.copy(gradients)
        
        # update x
        x[:-1] = prev_x[:-1] - gamma * gradients
        x[-1] = 1 - np.sum(x[:-1])  # the flow in the last path

        # if at least one of the flows is negative, change the gradients
        if np.sum(np.where(x < 0, 1, 0)) > 0:  # flow in at least one path is negtive
            gradients[x[:-1]< 0 ] =  prev_x[:-1][x[:-1]<0] / gamma
            x[:-1] = prev_x[:-1] - gamma * gradients
            x[-1] = 1 - np.sum(x[:-1])  # the flow in the last path
            # if the flow on the last path is still negative
            if x[-1] < 0:
                # define new gradients, increase negative ones
                # how much increase to make sure they are within the constraints?
                # reduce the amount proportional to the original
                gradients[gradients < 0] += (np.abs(x[-1]) / gamma) * (gradients[gradients<0] / gradients[gradients<0].sum())
                x[:-1] = prev_x[:-1] - gamma * gradients
                x[-1] = 1 - np.sum(x[:-1])  # the flow in the last path

        # update all flows, obj fun
        allflows = np.sum(path_arrays * x.reshape(num_variables, 1, 1), axis=0)
        obj_fun = affine_SO_obj(allflows, a, a0).sum()
        diff_value = obj_fun - prev_obj_fun
        print('Iteration %d: The total cost is %f, and the flow is ' % (k, obj_fun), x)

        # update gradients and step size
        gradients = np.array(
            [np.sum(  affine_cost(allflows, a, a0) * (
                        path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1, 0) *
                        path_arrays[-1]))
             for k in range(num_variables - 1)]
        ) + np.array(
            [np.sum(allflows * a * (
                        path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1, 0) *
                        path_arrays[-1]))
             for k in range(num_variables - 1)]
        )

        if np.where(np.abs(gradients - prev_gradients) < tol * np.abs(prev_gradients), 0, 1).sum() == 0:
            print('Wardrop equilibrium found:')
            print('Iteration %d: The total cost is %f, and the flow is ' % (k, obj_fun), x)
            G.SOflowsAffine[s][t] = x
            G.SOcostsAffine[s][t] = obj_fun
            G.SOflowsAffine_edge[s][t] = allflows
            return obj_fun, x

        # update step size
        gamma = np.inner(x[:-1] - prev_x[:-1], gradients - prev_gradients) / \
                np.inner(gradients - prev_gradients, gradients - prev_gradients)

    print('global minimum not found')
    return

