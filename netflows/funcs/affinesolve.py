# -*- coding: utf-8 -*-
"""
solve wardrop equilibrium or system optimal flow for affine cost functions a*t + a0
"""

from netflows.utils import affine_cost, affine_so_obj, we_affine_grad, so_affine_grad
import numpy as np
from tqdm import tqdm


def wardrop_equilibrium_affine_solve(G, s, t, tol=1e-12, maximum_iter=100000, cutoff=None, a=None, a0=None):
    """
    The function to solve Wardrop Equilibrium flow for a single source target pair under affine cost function setting.
    Usage:

    :param G: Graph object, storing the adjacency/weight/distance matrices
    :param s: source node
    :param t: target node
    :param tol: tolerance for convergence
    :param maximum_iter: maximum iteration times
    :param cutoff: a scalar value that defines maximal (binary) path length, namely,
        flows can only use paths shorter than the cutoff value
    :param a: the parameter of affine cost function, default is 1/weights
    :param a0: the parameter of affine cost function, default is distance
    :return: a tuple (x, allflows, total_cost_sum, total_cost).
        x: the vector storing flows on each path (path formulation)
        allflows: the matrix storing flows on each edge (edge formulation)
        total_cost_sum: total travel time for all flows from source to target
        total_cost: the matrix storing travel time on each edge (edge formulation)
    """

    # find all possible paths from s to t that are shorter than cutoff
    if cutoff is None:
        print("Cutoff not specified: shortest path distance + 1 taken as cutoff")
        cutoff = G.dijkstra(s, t) + 1 + 1
        if cutoff < 3:
            return
    # find all paths
    allpaths = G.findallpaths(s, t, cutoff)

    if a is None:
        a = G.rpl_weights

    if a0 is None:
        a0 = G.adj_dist

    return _wardrop_equilibrium_affine_solve(G, s, t, tol, maximum_iter, allpaths, a, a0)


def system_optimal_affine_solve(G, s, t, tol=1e-12, maximum_iter=100000, cutoff=None, a=None, a0=None):
    """
    The function to solve Wardrop Equilibrium flow for a single source target pair under affine cost function setting.
    Usage:

    :param G: Graph object, storing the adjacency/weight/distance matrices
    :param s: source node
    :param t: target node
    :param tol: tolerance for convergence
    :param maximum_iter: maximum iteration times
    :param cutoff: a scalar value that defines maximal (binary) path length, namely,
        flows can only use paths shorter than the cutoff value
    :param a: the parameter of affine cost function, default is 1/weights
    :param a0: the parameter of affine cost function, default is distance
    :return: a tuple (x, allflows, total_cost_sum, total_cost).
        x: the vector storing flows on each path (path formulation)
        allflows: the matrix storing flows on each edge (edge formulation)
        total_cost_sum: total travel time for all flows from source to target
        total_cost: the matrix storing travel time on each edge (edge formulation)
    """

    if cutoff is None:
        print("Cutoff not specified: take shortest path distance + 1 as cutoff")
        cutoff = G.dijkstra(s, t) + 1 + 1
        if cutoff < 3:
            return

    # find all paths
    allpaths = G.findallpaths(s, t, cutoff)

    if a is None:
        a = G.rpl_weights

    if a0 is None:
        a0 = G.adj_dist

    return _system_optimal_affine_solve(G, s, t, tol, maximum_iter, allpaths, a, a0)


def _wardrop_equilibrium_affine_solve(G, s, t, tol, maximum_iter, allpaths, a, a0):

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
    total_cost = allflows * affine_cost(allflows, a, a0)
    total_cost_sum = total_cost.sum()

    print('The initial flow (path formulation) is ', x)
    print('The initial cost is %f' % total_cost_sum)
    print('------------solving the Wardrop Equilibrium-------------')

    if num_variables < 2:
        print('Wardrop Equilibrium flow found:', x)
        print('The total travel time is %f' % total_cost_sum)
        return x, allflows, total_cost_sum, total_cost

    # initial gradients
    gradients = we_affine_grad(allflows, a, a0, path_arrays, num_variables)

    # initial estimation of step size gamma
    gamma1 = np.min(np.abs(x[:-1] / gradients))
    gamma2 = np.min(np.abs((1 - x[:-1]) / gradients))
    gamma = min(gamma1, gamma2) * 2 / 3  # to make sure the flow on each path is positive after initial iteration

    for k in tqdm(range(maximum_iter)):  # maximal iteration 10000
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
        total_cost = allflows * affine_cost(allflows, a, a0)
        total_cost_sum = total_cost.sum()
        
        # new gradients and stepsize
        gradients = we_affine_grad(allflows, a, a0, path_arrays, num_variables)
        
        if np.where(np.abs(gradients - prev_gradients) < tol, 0, 1).sum() == 0:  # convergence
            print('Wardrop Equilibrium flow found:', x)
            print('Iteration %d: the total travel time is %f' % (k, total_cost_sum))
            return x, allflows, total_cost_sum, total_cost

        # new step size
        gamma = np.inner(
            x[:-1] - prev_x[:-1], gradients - prev_gradients
        ) / np.inner(
            gradients - prev_gradients, gradients - prev_gradients
        )

    print('Wardrop Equilibrium flow not found')
    return


def _system_optimal_affine_solve(G, s, t, tol, maximum_iter, allpaths, a, a0):

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
    total_cost = allflows * affine_cost(allflows, a, a0)
    obj_fun = affine_so_obj(allflows, a, a0).sum()  # objective function is the total cost function

    print('The initial flow (path formulation) is ', x)
    print('The initial cost is %f' % obj_fun)
    print('------solve the system optimal flow------')

    if num_variables < 2:
        print('System Optimal flow found:', x)
        print('The total travel time is %f' % obj_fun)
        return x, allflows, obj_fun, total_cost

    # initial gradients
    gradients = so_affine_grad(allflows, a, a0, path_arrays, num_variables)

    # initial step size determination
    gamma1 = np.min(np.abs(x[:-1] / gradients))
    gamma2 = np.min(np.abs((1 - x[:-1]) / gradients))
    gamma = min(gamma1, gamma2) * 2 / 3

    for k in tqdm(range(maximum_iter)):  # maximal iteration 10000

        prev_x = np.copy(x)
        prev_gradients = np.copy(gradients)

        # update x
        x[:-1] = prev_x[:-1] - gamma * gradients
        x[-1] = 1 - np.sum(x[:-1])  # the flow in the last path
        # if at least one of the flows is negative, change the gradients
        if np.sum(np.where(x < 0, 1, 0)) > 0:  # flow in at least one path is negtive
            gradients[x[:-1] < 0] = prev_x[:-1][x[:-1] < 0] / gamma

            x[:-1] = prev_x[:-1] - gamma * gradients
            x[-1] = 1 - np.sum(x[:-1])  # the flow in the last path
            # if the flow on the last path is still negative
            if x[-1] < 0:
                # define new gradients, increase negative ones
                # how much increase to make sure they are within the constraints?
                # reduce the amount proportional to the original
                gradients[gradients < 0] += (
                                                    np.abs(x[-1]) / gamma
                                            ) * (
                        gradients[gradients < 0] / gradients[gradients < 0].sum()
                )
                x[:-1] = prev_x[:-1] - gamma * gradients
                x[-1] = 1 - np.sum(x[:-1])  # the flow in the last path

        # update all flows, obj fun
        allflows = np.sum(path_arrays * x.reshape(num_variables, 1, 1), axis=0)
        obj_fun = affine_so_obj(allflows, a, a0).sum()
        total_cost = allflows * affine_cost(allflows, a, a0)

        # update gradients and step size
        gradients = so_affine_grad(allflows, a, a0, path_arrays, num_variables)

        if np.where(np.abs(gradients - prev_gradients) < tol, 0, 1).sum() == 0:
            print('System Optimal flow found:', x)
            print('Iteration %d: the total travel time is %f' % (k, obj_fun))
            return x, allflows, obj_fun, total_cost

        # update step size
        gamma = np.inner(
            x[:-1] - prev_x[:-1], gradients - prev_gradients
        ) / np.inner(
            gradients - prev_gradients, gradients - prev_gradients
        )

    print('System Optimal flow not found')
    return
