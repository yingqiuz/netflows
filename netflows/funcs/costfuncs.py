
# coding: utf-8
# A list of cost functions: linear, affine, BPR, MM1

import numpy as np
from scipy.integrate import quad

##### Linear cost #####
def linear_cost(flow, weight):
    """linear cost function"""
    return flow * weight

def linear_integration(flow, weight):
    return quad(lambda x: linear_cost(x, weight), 0, flow)[0]

def linear_WE_obj(flow_mat, weight_mat):
    my_vec_integration = np.vectorize(linear_integration)
    return np.sum(my_vec_integration(flow_mat, weight_mat))

def linear_SO_obj(flow_mat, weight_mat):
    return np.sum(flow_mat * linear_cost(flow_mat, weight_mat))

##### Affine cost #####
def affine_cost(flow, weight, a0=0):
    """affine cost function"""
    return flow * weight + a0

def affine_integration(flow, weight, a0=0):
    return quad(lambda x: affine_cost(x, weight, a0), 0, flow)[0]

def affine_WE_obj(flow_mat, weight_mat, a0_mat = 0):
    my_vec_integration = np.vectorize(affine_integration)
    return np.sum(my_vec_integration(flow_mat, weight_mat, a0_mat))

def affine_SO_obj(flow_mat, weight_mat, a0_mat = 0):
    return np.sum(flow_mat * affine_cost(flow_mat, weight_mat, a0_mat))

def affine_we_obj_search(x, a, a0, path_arrays, num_variables):
    allflows = np.sum(path_arrays * np.append(x, 1 - np.sum(x)).reshape(num_variables, 1, 1), axis=0)
    my_vec_integration = np.vectorize(affine_integration)
    return np.sum(my_vec_integration(allflows, a, a0))

def affine_so_obj_search(x, a, a0, path_arrays, num_variables):
    allflows = np.sum(path_arrays * np.append(x, 1 - np.sum(x)).reshape(num_variables, 1, 1), axis=0)
    return np.sum(allflows * affine_cost(allflows, a, a0))

##### BPR cost #####
def BPR_cost(flow, a, u):
    """BPR cost function"""
    return a * (1 + 0.15 * (flow / u) ** 4)

def BPR_integration(flow, a, u):
    return quad(lambda x: BPR_cost(x, a, u), 0, flow)[0]

def BPR_WE_obj(flow_mat, a_mat, u_mat):
    my_vec_integration = np.vectorize(BPR_integration)
    return np.sum(my_vec_integration(flow_mat, a_mat, u_mat))

def BPR_SO_obj(flow_mat, a_mat, u_mat):
    return np.sum(flow_mat * BPR_cost(flow_mat, a_mat, u_mat))

##### MM1 cost #####
def MM1_cost(flow, weight):
    """MM1 cost function"""
    return 1 / (weight - flow)

def MM1_integration(flow, weight):
    return quad(lambda x : MM1_cost(x, weight), 0, flow)[0]

def MM1_WE_obj(flow_mat, weight_mat):
    my_vec_integration = np.vectorize(MM1_integration)
    return np.sum(my_vec_integration(flow_mat, weight_mat))

def MM1_SO_obj(flow_mat, weight_mat):
    return np.sum(flow_mat * MM1_cost(flow_mat, weight_mat))

