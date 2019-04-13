# -*- coding: utf-8 -*-
"""
solve wardrop equilibrium or system optimal flow for linear cost functions a*t
"""

from netflows.utils import (
    linear_cost, linear_so_obj, linear_we_obj,
    we_linear_grad, so_linear_grad
)
import numpy as np
from tqdm import tqdm


def wardrop_equilibrium_linear_solve(
        graph_object, s, t, tol=1e-8,
        maximum_iter=10000, cutoff=None, c=None
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
    :param c: the parameter of linear cost function, default is distance/weights
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

    if c is None:
        c = graph_object.dist_weight_ratio

    return _wardrop_equilibrium_linear_solve(
        graph_object, s, t, tol, maximum_iter, allpaths, c
    )


def system_optimal_linear_solve(
        graph_object, s, t, tol=1e-8,
        maximum_iter=10000, cutoff=None, c=None
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
    :param c: the parameter of linear cost function, default is distance/weights
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

    if c is None:
        c = graph_object.dist_weight_ratio

    return _system_optimal_linear_solve(
        graph_object, s, t, tol, maximum_iter, allpaths, c
    )


def _wardrop_equilibrium_linear_solve(
        graph_object, s, t, tol,
        maximum_iter, allpaths, c
):

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
        index_x = [path[k] for k in range(len(path) - 1)]  # x index of the adj matrix
        index_y = [path[k] for k in range(1, len(path))]  # y index of the adj matrix
        path_array_tmp[index_x, index_y] = 1
        path_arrays = np.append(path_arrays, path_array_tmp[np.newaxis, :], axis=0)

    # element (i, j) is the total flow on edge (i,j)
    allflows = np.sum(path_arrays * x.reshape(num_variables, 1, 1), axis=0)
    obj_fun = linear_we_obj(allflows, c).sum()
    total_cost = allflows * linear_cost(allflows, c)
    total_cost_sum = total_cost.sum()
    print('The initial flow is (path formulation) ', x)
    print('The initial cost is %f' % total_cost_sum)
    print('------------solving the Wardrop Equilibrium-------------')

    if num_variables < 2:
        print('Wardrop Equilibrium flow found:', x)
        print('The total travel time is %f' % total_cost_sum)
        return x, allflows, total_cost_sum, total_cost

    # new solver
    # get params for each parabola
    a = np.array(
        [((path_arrays[k] * (1 - path_arrays[-1]) +
           path_arrays[-1] * (1 - path_arrays[k])) * c).sum() / 2
         for k in range(num_variables - 1)]
    )
    for k in tqdm(range(maximum_iter)):
        prev_obj_fun = np.copy(obj_fun)
        # coordinates descent
        for kk in range(num_variables - 1):
            # get b
            b1 = (np.delete(
                path_arrays *
                (path_arrays[kk] * (1 - path_arrays[-1])) *
                x.reshape(num_variables, 1, 1),
                obj=[kk, num_variables - 1], axis=0
            ).sum(axis=0) * c).sum()
            b2 = -(np.delete(
                (1 - path_arrays) *
                (path_arrays[-1] * (1 - path_arrays[kk])) *
                x.reshape(num_variables, 1, 1),
                obj=[kk, num_variables-1], axis=0
            ).sum(axis=0) * c).sum()
            b = b1 + b2

            # compare 4 values
            boundary = np.array(
                [0, 1, -b / (2 * a), 1 - x[1:-1].sum()]
            )
            x[-1] -= np.sort(boundary)[1] - x[kk]
            x[kk] = np.sort(boundary)[1]
        allflows = np.sum(path_arrays * x.reshape(num_variables, 1, 1), axis=0)
        obj_fun = linear_we_obj(allflows, c).sum()
        if np.abs(obj_fun - prev_obj_fun) / prev_obj_fun < tol:
            total_cost = allflows * linear_cost(allflows, c)
            total_cost_sum = total_cost.sum()
            print('Wardrop Equilibrium flow found:', x)
            print('Iteration %d: the total travel time is %f' % (k, total_cost_sum))
            return x, allflows, total_cost_sum, total_cost

    print('Wardrop Equilibrium flow not found')
    return


def _system_optimal_linear_solve(
        graph_object, s, t, tol,
        maximum_iter, allpaths, c
):

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
    total_cost = allflows * linear_cost(allflows, c)
    obj_fun = linear_so_obj(allflows, c).sum()

    print('The initial flow (path formulation) is ', x)
    print('The initial cost is %f' % obj_fun)
    print('------solve the system optimal flow------')

    if num_variables < 2:
        print('System Optimal flow found:', x)
        print('The total travel time is %f' % obj_fun)
        return x, allflows, obj_fun, total_cost

        # new solver
        # get params for each parabola
    a = np.array(
        [((path_arrays[k] * (1 - path_arrays[-1]) +
           path_arrays[-1] * (1 - path_arrays[k])) * c).sum()
         for k in range(num_variables - 1)]
    )
    for k in tqdm(range(maximum_iter)):
        prev_obj_fun = np.copy(obj_fun)
        # coordinates descent
        for kk in range(num_variables - 1):
            # get b
            b1 = (np.delete(
                path_arrays *
                (path_arrays[kk] * (1 - path_arrays[-1])) *
                x.reshape(num_variables, 1, 1),
                obj=[kk, num_variables - 1], axis=0
            ).sum(axis=0) * c).sum() * 2
            b2 = -(np.delete(
                (1 - path_arrays) *
                (path_arrays[-1] * (1 - path_arrays[kk])) *
                x.reshape(num_variables, 1, 1),
                obj=[kk, num_variables - 1], axis=0
            ).sum(axis=0) * c).sum() * 2
            b = b1 + b2

            # compare 4 values
            boundary = np.array(
                [0, 1, -b / (2 * a), 1 - x[1:-1].sum()]
            )
            x[-1] -= np.sort(boundary)[1] - x[kk]
            x[kk] = np.sort(boundary)[1]
        allflows = np.sum(path_arrays * x.reshape(num_variables, 1, 1), axis=0)
        obj_fun = linear_so_obj(allflows, c).sum()
        if np.abs(obj_fun - prev_obj_fun) / prev_obj_fun < tol:
            total_cost = allflows * linear_cost(allflows, c)
            print('Wardrop Equilibrium flow found:', x)
            print('Iteration %d: the total travel time is %f' % (k, obj_fun))
            return x, allflows, obj_fun, total_cost

    print('Wardrop Equilibrium flow not found')
    return
