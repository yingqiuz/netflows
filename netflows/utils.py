# -*- coding: utf-8 -*-
"""
A list of cost functions (linear, affine, BPR, MM1) and the associated gradients
"""

import numpy as np
from scipy.integrate import quad


def linear_cost(flow, weight):
    """linear cost function"""
    return flow * weight


def linear_integration(flow, weight):
    return quad(lambda x: linear_cost(x, weight), 0, flow)[0]


def linear_we_obj(flow_mat, weight_mat):
    my_vec_integration = np.vectorize(linear_integration)
    return np.sum(my_vec_integration(flow_mat, weight_mat))


def linear_so_obj(flow_mat, weight_mat):
    return np.sum(flow_mat * linear_cost(flow_mat, weight_mat))


def affine_cost(flow, weight, a0):
    """affine cost function"""
    return flow * weight + a0


def affine_integration(flow, weight, a0):
    return quad(lambda x: affine_cost(x, weight, a0), 0, flow)[0]


def affine_we_obj(flow_mat, weight_mat, a0_mat):
    my_vec_integration = np.vectorize(affine_integration)
    return np.sum(my_vec_integration(flow_mat, weight_mat, a0_mat))


def affine_so_obj(flow_mat, weight_mat, a0_mat):
    return flow_mat * affine_cost(flow_mat, weight_mat, a0_mat)


def bpr_cost(flow, weight, u):
    """BPR cost function"""
    return weight * (1 + 0.15 * (flow * u) ** 4)


def bpr_integration(flow, weight, u):
    return quad(lambda x: bpr_cost(x, weight, u), 0, flow)[0]


def bpr_we_obj(flow_mat, weight_mat, u_mat ):
    my_vec_integration = np.vectorize(bpr_integration)
    return np.sum(my_vec_integration(flow_mat, weight_mat, u_mat))


def bpr_so_obj(flow_mat, weight_mat, u_mat ):
    return flow_mat * bpr_cost(flow_mat, weight_mat, u_mat)


def mm1_cost(flow, weight):
    """MM1 cost function"""
    return 1 / (weight - flow)


def mm1_integration(flow, weight):
    return quad(lambda x : mm1_cost(x, weight), 0, flow)[0]


def mm1_we_obj(flow_mat, weight_mat):
    my_vec_integration = np.vectorize(mm1_integration)
    return np.sum(my_vec_integration(flow_mat, weight_mat))


def mm1_so_obj(flow_mat, weight_mat):
    return np.sum(flow_mat * mm1_cost(flow_mat, weight_mat))


# gradients function
def we_affine_grad(x, a, a0, path_arrays, num_variables):
    allflows = np.sum(path_arrays * x.reshape(num_variables, 1, 1), axis = 0)
    gradients = np.array(
        [np.sum(affine_cost(allflows, a, a0) * path_arrays[k] )
         for k in range(num_variables - 1)]
    )
    return gradients


def so_affine_grad(x, a, a0, path_arrays, num_variables):
    allflows = np.sum(path_arrays * x.reshape(num_variables, 1, 1), axis=0)
    gradients = np.array(
        [np.sum(affine_cost(allflows, a, a0) *
                path_arrays[k] )
         for k in range(num_variables - 1)]
    ) + np.array(
        [np.sum(allflows * a *
                path_arrays[k])
         for k in range(num_variables - 1)]
    )
    return gradients


def we_linear_grad(x, a, path_arrays, num_variables):
    allflows = np.sum(path_arrays * x.reshape(num_variables, 1, 1), axis=0)
    gradients = np.array(
        [np.sum(linear_cost(allflows, a) * (path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) -
                                            np.where(path_arrays[k] == 0, 1, 0) * path_arrays[-1]))
         for k in range(num_variables - 1)]
    )
    return gradients


def so_linear_grad(x, a, path_arrays, num_variables):
    allflows = np.sum(path_arrays * x.reshape(num_variables, 1, 1), axis=0)
    gradients = np.array(
        [np.sum(linear_cost(allflows, a) * (
                path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1, 0) *
                path_arrays[-1]))
         for k in range(num_variables - 1)]
    ) + np.array(
        [np.sum(allflows * a * (
                path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1, 0) *
                path_arrays[-1]))
         for k in range(num_variables - 1)]
    )
    return gradients


def we_bpr_grad(x, a, u, path_arrays, num_variables):
    allflows = np.sum(path_arrays * np.append(x, 1 - np.sum(x)).reshape(num_variables, 1, 1), axis=0)
    gradients = np.array(
        [np.sum(bpr_cost(allflows, a, u) * (
                path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1, 0) *
                path_arrays[-1]))
         for k in range(num_variables - 1)]
    )
    return gradients


def so_bpr_grad(x, a, u, path_arrays, num_variables):
    allflows = np.sum(path_arrays * np.append(x, 1 - np.sum(x)).reshape(num_variables, 1, 1), axis=0)
    gradients = np.array(
        [np.sum(bpr_cost(allflows, a, u) * (
                path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1, 0) *
                path_arrays[-1]))
         for k in range(num_variables - 1)]
    ) + np.array(
        [np.sum(allflows * a * (0.6 * (allflows * u) ** 3 * u) * (
                path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1, 0) *
                path_arrays[-1]))
         for k in range(num_variables - 1)]
    )
    return gradients

