# -*- coding: utf-8 -*-
"""
solve wardrop equilibrium or system optimal flow for affine cost functions a*t + a0
"""

from netflows.utils import bpr_cost, bpr_so_obj, we_bpr_grad, so_bpr_grad
import numpy as np
from tqdm import tqdm


def wardrop_equilibrium_bpr_solve(G, s, t, tol=1e-8, maximum_iter=10000, cutoff=None, a=None, u=None):
    """
    The function to solve Wardrop Equilibrium flow for a single source target pair under BPR cost function setting.
    Usage:

    :param G: Graph object, storing the adjacency/weight/distance matrices
    :param s: source node
    :param t: target node
    :param tol: tolerance for convergence
    :param maximum_iter: maximum iteration times
    :param cutoff: a scalar value that defines maximal (binary) path length, namely,
        flows can only use paths shorter than the cutoff value
    :param a: the parameter of affine cost function, default is distance
    :param u: the parameter of affine cost function, default is 1/weight
    :return: a tuple (x, allflows, total_cost_sum, total_cost).
        x: the vector storing flows on each path (path formulation)
        allflows: the matrix storing flows on each edge (edge formulation)
        total_cost_sum: total travel time for all flows from source to target
        total_cost: the matrix storing travel time on each edge (edge formulation)
    """

    # find all possible paths from s to t that are shorter than cutoff
    if cutoff is None:
        print("Cutoff not specified: take shortest path distance + 1 as cutoff")
        cutoff = G.dijkstra(s, t) + 1 + 1
        if cutoff == 1:
            return
    allpaths = G.findallpaths(s, t, cutoff)

    if a is None:
        a = G.adj_dist

    # binarize dist matrix
    a[G.adj == 0] = 0

    if u is None:
        u = G.rpl_weights

    return _wardrop_equilibrium_bpr_solve(G, s, t, tol, maximum_iter, allpaths, a, u)


def system_optimal_bpr_solve(G, s, t, tol=1e-8, maximum_iter=10000, cutoff=None, a=None, u=None):
    """
    The function to solve System Optimal flow for a single source target pair under BPR cost function setting.
    Usage:

    :param G: Graph object, storing the adjacency/weight/distance matrices
    :param s: source node
    :param t: target node
    :param tol: tolerance for convergence
    :param maximum_iter: maximum iteration times
    :param cutoff: a scalar value that defines maximal (binary) path length, namely,
        flows can only use paths shorter than the cutoff value
    :param a: the parameter of affine cost function, default is distance
    :param u: the parameter of affine cost function, default is 1/weight
    :return: a tuple (x, allflows, total_cost_sum, total_cost).
        x: the vector storing flows on each path (path formulation)
        allflows: the matrix storing flows on each edge (edge formulation)
        total_cost_sum: total travel time for all flows from source to target
        total_cost: the matrix storing travel time on each edge (edge formulation)
    """

    # find all possible paths from s to t that are shorter than cutoff
    if cutoff is None:
        print("Cutoff not specified: take shortest path distance + 1 as cutoff")
        cutoff = G.dijkstra(s, t) + 1 + 1
        if cutoff == 1:
            return
    allpaths = G.findallpaths(s, t, cutoff)

    if a is None:
        a = G.adj_dist
    # binarize dist matrix
    a[G.adj == 0] = 0

    if u is None:
        u = G.rpl_weights

    return _system_optimal_bpr_solve(G, s, t, tol, maximum_iter, allpaths, a, u)


def _wardrop_equilibrium_bpr_solve(G, s, t, tol, maximum_iter, allpaths, a, u):

    num_variables = len(allpaths)  # the number of paths from s to t
    print('A total of %d paths found from %d to %d' % (num_variables, int(s), int(t)))

    # x is the flow vector (path formulation)
    x = np.ones((num_variables,)) / num_variables  # initial value

    # find equilibrium -- convex optimization
    # map to matrix
    path_arrays = np.empty((0, G.adj.shape[0], G.adj.shape[1]))  # list of matrix to store path flows
    print('constructing edge formulations...')
    for path in tqdm(allpaths, total=num_variables):
        path_array_tmp = np.zeros(G.adj.shape)
        index_x = [path[k] for k in range(len(path) - 1)]  # x index of the adj matrix
        index_y = [path[k] for k in range(1, len(path))]  # y index of the adj matrix
        path_array_tmp[index_x, index_y] = 1
        path_arrays = np.append(path_arrays, path_array_tmp[np.newaxis, :], axis=0)

    # element (i, j) is the total flow on edge (i,j)    
    allflows = np.sum(path_arrays * x.reshape(num_variables, 1, 1), axis=0)
    allflows[G.adj == 0] = 0

    total_cost = allflows * bpr_cost(allflows, a, u)
    total_cost_sum = total_cost.sum()

    print('The initial flow (path formulation) is ', x)
    print('The initial cost is %f' % total_cost_sum)
    print('------------solving the Wardrop Equilibrium-------------')

    if num_variables < 2:
        print('Wardrop Equilibrium flow found:', x)
        print('The total travel time is %f' % total_cost_sum)
        return x, allflows, total_cost_sum, total_cost

    gradients = we_bpr_grad(x, a, u, path_arrays, num_variables)
    
    # initial step size determination
    gamma1 = np.min(np.abs(x[:-1] / gradients))
    gamma2 = np.min(np.abs((1 - x[:-1]) / gradients))
    gamma = min(gamma1, gamma2) * 2 / 3

    for k in range(maximum_iter):  # maximal iteration 10000

        prev_x = np.copy(x)
        prev_gradients = np.copy(gradients)

        # update x
        x[:-1] = prev_x[:-1] - gamma * gradients
        x[-1] = 1 - np.sum(x[:-1])  # the flow in the last path

        if np.sum(np.where(x < 0, 1, 0)) > 0:  # flow in at least one path is negtive
            # re-adapt the gradients
            gradients[x[:-1] < 0] = prev_x[:-1][x[:-1] < 0] / gamma
            x[:-1] = prev_x[:-1] - gamma * gradients
            x[-1] = 1 - np.sum(x[:-1])  # the flow in the last path
            # print('new x is ', x)
            if x[-1] < 0:
                # define new gradients, increase negative ones
                # how much increase to make sure they are within the constraints?
                # reduce the amount proportional to the original
                gradients[gradients < 0] += (np.abs(x[-1]) / gamma) * (
                        gradients[gradients < 0] / gradients[gradients < 0].sum())
                x[:-1] = prev_x[:-1] - gamma * gradients
                x[-1] = 1 - np.sum(x[:-1])  # the flow in the last path
                  
        # update allflows and costs
        allflows = np.sum(path_arrays * x.reshape(num_variables, 1, 1), axis=0)

        total_cost = allflows * bpr_cost(allflows, a, u)
        total_cost_sum = total_cost.sum()

        # new gradients
        gradients = we_bpr_grad(x, a, u, path_arrays, num_variables)
        
        if np.sum(np.where(np.abs(gradients-prev_gradients) < tol, 0, 1)) == 0:  # test convergence
            print('Wardrop Equilibrium flow found:', x)
            print('Iteration %d: the total travel time is %f' % (k, total_cost_sum))
            return x, allflows, total_cost_sum, total_cost

        gamma = np.inner(
            x[:-1] - prev_x[:-1], gradients - prev_gradients
        ) / np.inner(
            gradients - prev_gradients, gradients - prev_gradients
        )

    print('Wardrop Equilibrium flow not found')
    return


def _system_optimal_bpr_solve(G, s, t, tol, maximum_iter, allpaths, a, u):

    num_variables = len(allpaths)  # the number of paths from s to t
    print('A total of %d paths found from %d to %d' % (num_variables, int(s), int(t)))

    # x is the flow vector (path formulation)
    x = np.ones((num_variables,)) / num_variables  # initial value

    # find equilibrium -- convex optimization
    # map to matrix
    path_arrays = np.empty((0,   G.adj.shape[0],   G.adj.shape[1]))  # list of matrix to store path flows
    print('constructing edge formulations...')
    for path in tqdm(allpaths, total=num_variables):
        path_array_tmp = np.zeros(G.adj.shape)
        index_x = [path[k] for k in range(len(path) - 1)]  # x index of the adj matrix
        index_y = [path[k] for k in range(1, len(path))]  # y index of the adj matrix
        path_array_tmp[index_x, index_y] = 1
        path_arrays = np.append(path_arrays, path_array_tmp[np.newaxis, :], axis=0)

    # element (i, j) is the total flow on edge (i,j)    
    allflows = np.sum(path_arrays * x.reshape(num_variables, 1, 1), axis=0)
    total_cost = allflows * bpr_cost(allflows, a, u)
    obj_fun = bpr_so_obj(allflows, a, u).sum()  # objective function is the total cost

    print('The initial flow (path formulation) is ', x)
    print('The initial cost is %f' % obj_fun)
    print('------solve the system optimal flow------')

    if num_variables < 2:
        print('System Optimal flow found:', x)
        print('The total travel time is %f' % obj_fun)
        return x, allflows, obj_fun, total_cost

    gradients = so_bpr_grad(x, a, u, path_arrays, num_variables)

    # initial step size
    gamma1 = np.min(np.abs(x[:-1] / gradients))
    gamma2 = np.min(np.abs((1 - x[:-1]) / gradients))
    gamma = min(gamma1, gamma2) * 2 / 3

    for k in range(maximum_iter):  # maximal iteration 10000
                  
        prev_x = np.copy(x)
        prev_gradients = np.copy(gradients)

        # update x
        x[:-1] = prev_x[:-1] - gamma * gradients
        x[-1] = 1 - np.sum(x[:-1])  # the flow in the last path

        if np.sum(np.where(x < 0, 1, 0)) > 0:  # flow in at least one path is negtive
            gradients[x[:-1] < 0] = prev_x[:-1][x[:-1] < 0] / gamma
            x[:-1] = prev_x[:-1] - gamma * gradients
            x[-1] = 1 - np.sum(x[:-1])  # the flow in the last path
            if x[-1] < 0:
                # define new gradients, increase negative ones
                # how much increase to make sure they are within the constraints?
                # reduce the amount proportional to the original
                gradients[gradients < 0] += (np.abs(x[-1]) / gamma) * (
                        gradients[gradients < 0] / gradients[gradients < 0].sum())
                x[:-1] = prev_x[:-1] - gamma * gradients
                x[-1] = 1 - np.sum(x[:-1])  # the flow in the last path

        allflows = np.sum(path_arrays * x.reshape(num_variables, 1, 1), axis=0)
        obj_fun = bpr_so_obj(allflows, a, u).sum()
        total_cost = allflows * bpr_cost(allflows, a, u)

        # update gradients
        gradients = so_bpr_grad(x, a, u, path_arrays, num_variables)

        # convergence?
        if np.sum(np.where(np.abs(gradients-prev_gradients) < tol, 0, 1)) == 0:
            print('System Optimal flow found:', x)
            print('Iteration %d: the total travel time is %f' % (k, obj_fun))
            return x, allflows, obj_fun, total_cost
                  
        # new step size
        gamma = np.inner(
            x[:-1] - prev_x[:-1], gradients - prev_gradients
        ) / np.inner(
            gradients - prev_gradients, gradients - prev_gradients
        )

    print('global minimum not found')
    return
