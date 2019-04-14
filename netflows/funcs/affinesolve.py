# -*- coding: utf-8 -*-
"""
solve wardrop equilibrium or system optimal flow for affine cost functions a*t + a0
"""

from netflows.utils import affine_cost, affine_so_obj, affine_we_obj

import numpy as np
from tqdm import tqdm


def wardrop_equilibrium_affine_solve(
        graph_object, s, t, tol=1e-12,
        maximum_iter=100000, cutoff=None, c=None, c0=None
):
    """
    The function to solve Wardrop Equilibrium flow for a single source target pair
    under affine cost function setting.
    Usage:

    :param graph_object: Graph object, storing the adjacency/weight/distance matrices
    :param s: source node
    :param t: target node
    :param tol: tolerance for convergence
    :param maximum_iter: maximum iteration times
    :param cutoff: a scalar value that defines maximal (binary) path length, namely,
        flows can only use paths shorter than the cutoff value
    :param c: the parameter of affine cost function, default is 1/weights
    :param c0: the parameter of affine cost function, default is distance
    :return: a tuple (x, allflows, total_cost_sum, total_cost).
        x: the vector storing flows on each path (path formulation)
        allflows: the matrix storing flows on each edge (edge formulation)
        total_cost_sum: total travel time for all flows from source to target
        total_cost: the matrix storing travel time on each edge (edge formulation)
    """

    # find all possible paths from s to t that are shorter than cutoff
    if cutoff is None:
        print("Cutoff not specified: shortest path distance + 1 taken as cutoff")
        cutoff = graph_object.dijkstra(s, t) + 1 + 1
        if cutoff < 3:
            return
    # find all paths
    allpaths = graph_object.findallpaths(s, t, cutoff)

    if c is None:
        c = graph_object.rpl_weights

    if c0 is None:
        c0 = graph_object.adj_dist

    return _wardrop_equilibrium_affine_solve(
        graph_object, s, t, tol, maximum_iter, allpaths, c, c0
    )


def system_optimal_affine_solve(
        graph_object, s, t, tol=1e-12,
        maximum_iter=100000, cutoff=None, c=None, c0=None
):
    """
    The function to solve Wardrop Equilibrium flow for a single source target pair
    under affine cost function setting.
    Usage:

    :param graph_object: Graph object, storing the adjacency/weight/distance matrices
    :param s: source node
    :param t: target node
    :param tol: tolerance for convergence
    :param maximum_iter: maximum iteration times
    :param cutoff: a scalar value that defines maximal (binary) path length, namely,
        flows can only use paths shorter than the cutoff value
    :param c: the parameter of affine cost function, default is 1/weights
    :param c0: the parameter of affine cost function, default is distance
    :return: a tuple (x, allflows, total_cost_sum, total_cost).
        x: the vector storing flows on each path (path formulation)
        allflows: the matrix storing flows on each edge (edge formulation)
        total_cost_sum: total travel time for all flows from source to target
        total_cost: the matrix storing travel time on each edge (edge formulation)
    """

    if cutoff is None:
        print("Cutoff not specified: take shortest path distance + 1 as cutoff")
        cutoff = graph_object.dijkstra(s, t) + 1 + 1
        if cutoff < 3:
            return

    # find all paths
    allpaths = graph_object.findallpaths(s, t, cutoff)

    if c is None:
        c = graph_object.rpl_weights

    if c0 is None:
        c0 = graph_object.adj_dist

    return _system_optimal_affine_solve(
        graph_object, s, t, tol, maximum_iter, allpaths, c, c0
    )


def _wardrop_equilibrium_affine_solve(
        graph_object, s, t, tol, maximum_iter, allpaths, c, c0
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
    obj_fun = affine_we_obj(allflows, c, c0).sum()
    total_cost = allflows * affine_cost(allflows, c, c0)
    total_cost_sum = total_cost.sum()

    print('The initial flow (path formulation) is ', x)
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
    b0 = np.array(
        [((path_arrays[k] * (1 - path_arrays[-1]) -
           path_arrays[-1] * (1 - path_arrays[k])) * c0).sum()
         for k in range(num_variables - 1)]
    )
    for k in tqdm(range(maximum_iter)):
        prev_x = np.copy(x)
        # coordinates descent
        for kk in range(num_variables - 1):
            # get b
            b1 = (np.delete(
                path_arrays *
                (path_arrays[kk] * (1 - path_arrays[-1])) *
                x.reshape(num_variables, 1, 1),
                obj=[kk, num_variables - 1], axis=0
            ).sum(axis=0) * c).sum()
            b2 = -(
                    (np.delete(
                        path_arrays * (path_arrays[-1] * (1 - path_arrays[kk])) *
                        x.reshape(num_variables, 1, 1),
                        obj=[kk, num_variables - 1], axis=0
                    ).sum(axis=0) + (
                            path_arrays[-1] * (1 - path_arrays[kk]) *
                            (1 - x[:-1].sum() + x[kk]))
                     ) * c
            ).sum()
            b = b1 + b2 + b0[kk]

            # compare 4 values
            boundary = np.array(
                [0, 1, -b / (2 * a[kk]), 1 - x[1:-1].sum()]
            )
            x[-1] -= np.sort(boundary)[1] - x[kk]
            x[kk] = np.sort(boundary)[1]

        allflows = np.sum(path_arrays * x.reshape(num_variables, 1, 1), axis=0)
        obj_fun = affine_we_obj(allflows, c, c0).sum()
        if np.where(np.abs(x - prev_x) < tol * prev_x, 0, 1).sum == 0:
            total_cost = allflows * affine_cost(allflows, c, c0)
            total_cost_sum = total_cost.sum()
            print('Wardrop Equilibrium flow found:', x)
            print('Iteration %d: the total travel time is %f' % (k, total_cost_sum))
            return x, allflows, total_cost_sum, total_cost

    print('Wardrop Equilibrium flow not found')
    return


def _system_optimal_affine_solve(
        graph_object, s, t, tol, maximum_iter, allpaths, c, c0
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
    total_cost = allflows * affine_cost(allflows, c, c0)
    # objective function is the total cost function
    obj_fun = affine_so_obj(allflows, c, c0).sum()

    print('The initial flow (path formulation) is ', x)
    print('The initial cost is %f' % obj_fun)
    print('------solve the system optimal flow------')

    if num_variables < 2:
        print('System Optimal flow found:', x)
        print('The total travel time is %f' % obj_fun)
        return x, allflows, obj_fun, total_cost

    a = np.array(
        [((path_arrays[k] * (1 - path_arrays[-1]) +
           path_arrays[-1] * (1 - path_arrays[k])) * c).sum()
         for k in range(num_variables - 1)]
    )
    b0 = np.array(
        [((path_arrays[k] * (1 - path_arrays[-1]) -
           path_arrays[-1] * (1 - path_arrays[k])) * c0).sum()
         for k in range(num_variables - 1)]
    )

    for k in tqdm(range(maximum_iter)):
        prev_x = np.copy(x)
        # coordinates descent
        for kk in range(num_variables - 1):
            # get b
            b1 = (np.delete(
                path_arrays *
                (path_arrays[kk] * (1 - path_arrays[-1])) *
                x.reshape(num_variables, 1, 1),
                obj=[kk, num_variables - 1], axis=0
            ).sum(axis=0) * c).sum() * 2
            b2 = -(
                    (np.delete(
                        path_arrays * (path_arrays[-1] * (1 - path_arrays[kk])) *
                        x.reshape(num_variables, 1, 1),
                        obj=[kk, num_variables - 1], axis=0
                    ).sum(axis=0) + (
                            path_arrays[-1] * (1 - path_arrays[kk]) *
                            (1 - x[:-1].sum() + x[kk]))
                     ) * c
            ).sum() * 2
            b = b1 + b2 + b0[kk]

            # compare 4 values
            boundary = np.array(
                [0, 1, -b / (2 * a[kk]), 1 - x[1:-1].sum()]
            )
            x[-1] -= np.sort(boundary)[1] - x[kk]
            x[kk] = np.sort(boundary)[1]
        allflows = np.sum(path_arrays * x.reshape(num_variables, 1, 1), axis=0)
        obj_fun = affine_so_obj(allflows, c, c0).sum()
        if np.where(np.abs(x - prev_x) < tol * prev_x, 0, 1).sum == 0:
            total_cost = allflows * affine_cost(allflows, c, c0)
            print('Wardrop Equilibrium flow found:', x)
            print('Iteration %d: the total travel time is %f' % (k, obj_fun))
            return x, allflows, obj_fun, total_cost

    print('Wardrop Equilibrium flow not found')
    return
