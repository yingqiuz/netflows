# -*- coding: utf-8 -*-

import numpy as np
import pytest
from netflows import create_graph, wardrop_equilibrium_affine_solve, system_optimal_affine_solve

ADJ_MAT = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
DIST_MAT = np.array([[0, 0, 1, 0], [0, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0]])
WEIGHT_MAT = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1/2], [0, 0, 0, 0]])

WE_FLOW = np.array([0.4, 0.2, 0.4])
SO_FLOW = np.array([0, 0.5, 0.5])

WE_COST = 13 / 5
SO_COST = 9 / 4

S = 0
T = 3


@pytest.fixture
def test_graph():
    G = create_graph(adj=ADJ_MAT, dist=DIST_MAT, weights=WEIGHT_MAT)
    return G


def test_we_affine(test_graph):
    x, allflows, total_cost_sum, total_cost = wardrop_equilibrium_affine_solve(
        test_graph, S, T, tol=1e-8, maximum_iter=10000
    )
    errors = []

    if not np.allclose(x, WE_FLOW, rtol=1e-8):
        errors.append("WE flow is incorrect")

    if not np.abs(total_cost_sum - WE_COST) < WE_COST * 1e-8:
        errors.append("WE travel time is incorrect")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))


def test_so_affine(test_graph):
    x, allflows, obj_fun, total_cost = system_optimal_affine_solve(
        test_graph, S, T, tol=1e-8, maximum_iter=10000
    )
    errors = []

    if not np.allclose(x, SO_FLOW, rtol=1e-8):
        errors.append("SO flow is incorrect")

    if not np.abs(obj_fun - SO_COST) < SO_COST * 1e-8:
        errors.append("SO travel time is incorrect")

    assert not errors, "errors occurred:\n{}".format("\n".join(errors))
