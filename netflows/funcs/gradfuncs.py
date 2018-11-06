from __future__ import absolute_import

import numpy as np
from netflows.funcs.costfuncs import *

def we_affine_grad(x, a, a0, path_arrays, num_variables):
    allflows = np.sum(path_arrays * np.append(x, 1 - np.sum(x)).reshape(num_variables, 1, 1), axis = 0)
    gradients = np.array(
        [np.sum(linear_cost(allflows, a) * (path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) -
                                            np.where(path_arrays[k] == 0, 1, 0) * path_arrays[-1]))
         for k in range(num_variables - 1)]
    )
    return gradients

def so_affine_grad(x, a, a0, path_arrays, num_variables):
    allflows = np.sum(path_arrays * np.append(x, 1 - np.sum(x)).reshape(num_variables, 1, 1), axis=0)
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

def we_linear_grad(x, a, path_arrays, num_variables):
    allflows = np.sum(path_arrays * np.append(x, 1 - np.sum(x)).reshape(num_variables, 1, 1), axis=0)
    gradients = np.array(
        [np.sum(linear_cost(allflows, a) * (path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) -
                                            np.where(path_arrays[k] == 0, 1, 0) * path_arrays[-1]))
         for k in range(num_variables - 1)]
    )
    return gradients

def so_linear_grad(x, a, path_arrays, num_variables):
    allflows = np.sum(path_arrays * np.append(x, 1 - np.sum(x)).reshape(num_variables, 1, 1), axis=0)
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
        [np.sum(BPR_cost(allflows, a, u) * (
                path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1, 0) *
                path_arrays[-1]))
         for k in range(num_variables - 1)]
    )
    return gradients

def so_bpr_grad(x, a, u, path_arrays, num_variables):
    allflows = np.sum(path_arrays * np.append(x, 1 - np.sum(x)).reshape(num_variables, 1, 1), axis=0)
    gradients = np.array(
        [np.sum(BPR_cost(allflows, a, u) * (
                path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1, 0) *
                path_arrays[-1]))
         for k in range(num_variables - 1)]
    ) + np.array(
        [np.sum(allflows * a * (0.6 * (allflows / u) ** 3 / u) * (
                path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1, 0) *
                path_arrays[-1]))
         for k in range(num_variables - 1)]
    )
    return gradients