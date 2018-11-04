
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
    return my_vec_integration(flow_mat, weight_mat)

def linear_SO_obj(flow_mat, weight_mat):
    return flow_mat * linear_cost(flow_mat, weight_mat)

##### Affine cost #####
def affine_cost(flow, weight, a0=0):
    """affine cost function"""
    return flow * weight + a0

def affine_integration(flow, weight, a0=0):
    return quad(lambda x: affine_cost(x, weight, a0), 0, flow)[0]

def affine_WE_obj(flow_mat, weight_mat, a0_mat = 0):
    my_vec_integration = np.vectorize(affine_integration)
    return my_vec_integration(flow_mat, weight_mat, a0_mat)

def affine_SO_obj(flow_mat, weight_mat, a0_mat = 0):
    return flow_mat * affine_cost(flow_mat, weight_mat, a0_mat)

##### BPR cost #####
def BPR_cost(flow, weight, u = 1):
    """BPR cost function"""
    return weight * (1 + 0.15 * (flow / u) ** 4)

def BPR_integration(flow, weight, u=1):
    return quad(lambda x: BPR_cost(x, weight, u), 0, flow)[0]

def BPR_WE_obj(flow_mat, weight_mat, u_mat = 1):
    my_vec_integration = np.vectorize(BPR_integration)
    return my_vec_integration(flow_mat, weight_mat, u_mat)

def BPR_SO_obj(flow_mat, weight_mat, u_mat = 1):
    return flow_mat * BPR_cost(flow_mat, weight_mat, u_mat)

##### MM1 cost #####
def MM1_cost(flow, weight):
    """MM1 cost function"""
    return 1 / (weight - flow)

def MM1_integration(flow, weight):
    return quad(lambda x : MM1_cost(x, weight), 0, flow)[0]

def MM1_WE_obj(flow_mat, weight_mat):
    my_vec_integration = np.vectorize(MM1_integration)
    return my_vec_integration(flow_mat, weight_mat)

def MM1_SO_obj(flow_mat, weight_mat):
    return flow_mat * MM1_cost(flow_mat, weight_mat)

