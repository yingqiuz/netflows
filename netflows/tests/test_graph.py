# -*- coding: utf-8 -*-

import numpy as np
import pytest
from netflows import Graph

ADJ_MAT = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
DIST_MAT = np.array([[0, 0, 1, 0], [0, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0]])
WEIGHT_MAT = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1/2], [0, 0, 0, 0]])


@pytest.fixture
def test_graph():
    G = Graph(adj=ADJ_MAT, dist=DIST_MAT, weights=WEIGHT_MAT)
    return G


def test_adj(test_graph):
    assert test_graph.adj == ADJ_MAT


def test_weight_adj(test_graph):
    assert test_graph.adj_weights == ADJ_MAT * WEIGHT_MAT


def test_dist_adj(test_graph):
    assert test_graph.adj_dist == ADJ_MAT * DIST_MAT


def test_find_all_paths(test_graph):
    allpaths = test_graph.findallpaths(0, 3)
    assert len(allpaths) == 3


def test_dijkstra(test_graph):
    d = test_graph.dijkstra(0, 3)
    assert d == 2

