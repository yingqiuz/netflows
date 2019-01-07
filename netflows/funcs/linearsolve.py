# -*- coding: utf-8 -*-
"""
solve wardrop equilibrium or system optimal flow for linear cost functions a*t
"""

from netflows.utils import linear_cost, linear_so_obj, we_linear_grad, so_linear_grad
import numpy as np
from tqdm import tqdm


def wardrop_equilibrium_linear_solve(
        graph_object, s, t, tol=1e-8, maximum_iter=10000, cutoff=None, a=None
):
    """
    The function to solve Wardrop Equilibrium flow for a single source target pair
    under linear cost function setting.
    Usage:

    :param graph_object: Graph object, storing the adjacency/weight/distance matrices
    :param s: source node
    :param t: target node
    :param tol: tolerance for convergence
    :param maximum_iter: maximum iteration times
    :param cutoff: a scalar value that defines maximal (binary) path length, namely,
        flows can only use paths shorter than the cutoff value
    :param a: the parameter of linear cost function, default is distance/weights
    :return: a tuple (x, allflows, total_cost_sum, total_cost).
        x: the vector storing flows on each path (path formulation)
        allflows: the matrix storing flows on each edge (edge formulation)
        total_cost_sum: total travel time for all flows from source to target
        total_cost: the matrix storing travel time on each edge (edge formulation)
    """

    # find all possible paths from s to t that are shorter than cutoff
    if cutoff is None:
        print("Cutoff is not specified: shortest path distance + 1 taken as cutoff")
        cutoff = graph_object.dijkstra(s, t) + 1 + 1
        if cutoff < 3:
            return

    allpaths = graph_object.findallpaths(s, t, cutoff)

    if a is None:
        a = graph_object.dist_weight_ratio

    return _wardrop_equilibrium_linear_solve(
        graph_object, s, t, tol, maximum_iter, allpaths, a
    )


def system_optimal_linear_solve(
        graph_object, s, t, tol=1e-8, maximum_iter=10000, cutoff=None, a=None
):
    """
    The function to solve system optimal flow for a single source target pair
    under linear cost function setting.
    Usage:

    :param graph_object: Graph object, storing the adjacency/weight/distance matrices
    :param s: source node
    :param t: target node
    :param tol: tolerance for convergence
    :param maximum_iter: maximum iteration times
    :param cutoff: a scalar value that defines maximal (binary) path length, namely,
        flows can only use paths shorter than the cutoff value
    :param a: the parameter of linear cost function, default is distance/weights
    :return: a tuple (x, allflows, total_cost_sum, total_cost).
        x: the vector storing flows on each path (path formulation)
        allflows: the matrix storing flows on each edge (edge formulation)
        total_cost_sum: total travel time for all flows from source to target
        total_cost: the matrix storing travel time on each edge (edge formulation)
    """

    # find all possible paths from s to t that are shorter than cutoff
    if cutoff is None:
        print("Cutoff not specified: take shortest path distance + 1 as cutoff")
        cutoff = graph_object.dijkstra(s, t) + 1 + 1
        if cutoff < 3:
            return

    allpaths = graph_object.findallpaths(s, t, cutoff)

    if a is None:
        a = graph_object.dist_weight_ratio

    return _system_optimal_linear_solve(
        graph_object, s, t, tol, maximum_iter, allpaths, a
    )


def _wardrop_equilibrium_linear_solve(graph_object, s, t, tol, maximum_iter, allpaths, a):

    num_variables = len(allpaths)  # the number of paths from s to t
    print('A total of %d paths found from %d to %d' % (num_variables, int(s), int(t)))

    # x is the flow vector (path formulation)
    x = np.ones((num_variables, )) / num_variables  # initial value

    # find equilibrium -- convex optimization
    # map paths to matrix
    path_arrays = np.empty((0, graph_object.adj.shape[0], graph_object.adj.shape[1]))
    print('constructing edge formulations...')
    for path in tqdm(allpaths, total=num_variables):
        path_array_tmp = np.zeros(graph_object.adj.shape)
        index_x = [path[k] for k in range(len(path)-1)]  # x index of the adj matrix
        index_y = [path[k] for k in range(1, len(path))]  # y index of the adj matrix
        path_array_tmp[index_x, index_y] = 1
        path_arrays = np.append(path_arrays, path_array_tmp[np.newaxis, :], axis=0)

    # element (i, j) is the total flow on edge (i,j)
    allflows = np.sum(path_arrays * x.reshape(num_variables, 1, 1), axis=0)
    total_cost = allflows * linear_cost(allflows, a)
    total_cost_sum = total_cost.sum()
    print('The initial flow is (path formulation) ', x)
    print('The initial cost is %f' % total_cost_sum)
    print('------------solving the Wardrop Equilibrium-------------')

    if num_variables < 2:
        print('Wardrop Equilibrium flow found:', x)
        print('The total travel time is %f' % total_cost_sum)
        return x, allflows, total_cost_sum, total_cost

    gradients = we_linear_grad(allflows, a, path_arrays, num_variables)

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

        if np.sum(np.where(x < 0, 1, 0)) > 0:
            # flow in at least one path is negative
            gradients[x[:-1] < 0] = prev_x[:-1][x[:-1] < 0] / gamma
            x[:-1] = prev_x[:-1] - gamma * gradients
            x[-1] = 1 - np.sum(x[:-1])  # the flow in the last path
            # the flow on the last path is still negative
            if x[-1] < 0:
                # define new gradients, increase negative ones
                # how much increase to make sure they are within the constraints?
                # reduce the amount proportional to the original
                gradients[gradients < 0] += (np.abs(x[-1]) / gamma) * (
                        gradients[gradients < 0] / gradients[gradients < 0].sum())
                x[:-1] = prev_x[:-1] - gamma * gradients
                x[-1] = 1 - np.sum(x[:-1])  # the flow in the last path

        # update allflows, obj func
        allflows = np.sum(path_arrays * x.reshape(num_variables, 1, 1), axis=0)
        total_cost = allflows * linear_cost(allflows, a)
        total_cost_sum = total_cost.sum()

        # new gradients and step size
        gradients = we_linear_grad(allflows, a, path_arrays, num_variables)

        if np.sum(np.where(np.abs(gradients-prev_gradients) < tol, 0, 1)) == 0:
            print('Wardrop Equilibrium flow found:', x)
            print('Iteration %d: the total travel time is %f' % (k, total_cost_sum))
            return x, allflows, total_cost_sum, total_cost

        # update gamma
        gamma = np.inner(
            x[:-1] - prev_x[:-1], gradients - prev_gradients
        ) / np.inner(
            gradients - prev_gradients, gradients - prev_gradients
        )

    print('Wardrop Equilibrium flow not found')
    return


def _system_optimal_linear_solve(graph_object, s, t, tol, maximum_iter, allpaths, a):

    num_variables = len(allpaths)  # the number of paths from s to t
    print('A total of %d paths found from %d to %d' % (num_variables, int(s), int(t)))

    # x is the flow vector (path formulation)
    x = np.ones((num_variables,)) / num_variables  # initial value

    # find equilibrium -- convex optimization
    # map to matrix
    path_arrays = np.empty((0, graph_object.adj.shape[0], graph_object.adj.shape[1]))
    print('constructing edge formulations...')
    for path in tqdm(allpaths, total=num_variables):
        path_array_tmp = np.zeros(graph_object.adj.shape)
        index_x = [path[k] for k in range(len(path) - 1)]  # x index of the adj matrix
        index_y = [path[k] for k in range(1, len(path))]  # y index of the adj matrix
        path_array_tmp[index_x, index_y] = 1
        path_arrays = np.append(path_arrays, path_array_tmp[np.newaxis, :], axis=0)

    # element (i, j) is the total flow on edge (i,j)
    allflows = np.sum(path_arrays * x.reshape(num_variables, 1, 1), axis=0)
    total_cost = allflows * linear_cost(allflows, a)
    obj_fun = linear_so_obj(allflows, a).sum()

    print('The initial flow (path formulation) is ', x)
    print('The initial cost is %f' % obj_fun)
    print('------solve the system optimal flow------')

    if num_variables < 2:
        print('System Optimal flow found:', x)
        print('The total travel time is %f' % obj_fun)
        return x, allflows, obj_fun, total_cost

    gradients = so_linear_grad(allflows, a, path_arrays, num_variables)

    # initial step size determination
    gamma1 = np.min(np.abs(x[:-1]/gradients))
    gamma2 = np.min(np.abs((1 - x[:-1]) / gradients))
    gamma = min(gamma1, gamma2) * 2 / 3

    for k in tqdm(range(maximum_iter)):  # maximal iteration 10000
        prev_x = np.copy(x)
        prev_gradients = np.copy(gradients)

        # update x
        x[:-1] = prev_x[:-1] - gamma * gradients
        x[-1] = 1 - np.sum(x[:-1])  # the flow in the last path

        if np.sum(np.where(x < 0, 1, 0)) > 0:  # flow in at least one path is negtive
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

        allflows = np.sum(path_arrays * x.reshape(num_variables, 1, 1), axis=0)
        total_cost = allflows * linear_cost(allflows, a)
        obj_fun = linear_so_obj(allflows, a).sum()

        # new gradients and gamma
        gradients = so_linear_grad(allflows, a, path_arrays, num_variables)

        if np.sum(np.where(np.abs(gradients-prev_gradients) < tol, 0, 1)) == 0:
            print('System Optimal flow found:', x)
            print('Iteration %d: the total travel time is %f' % (k, obj_fun))
            return x, allflows, obj_fun, total_cost

        gamma = np.inner(
            x[:-1] - prev_x[:-1], gradients - prev_gradients
        ) / np.inner(
            gradients - prev_gradients, gradients - prev_gradients
        )

    print('System Optimal flow not found')
    return
