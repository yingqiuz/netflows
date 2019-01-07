# -*- coding: utf-8 -*-

import numpy as np
from heapq import heappop, heappush


class CreateGraph:
    def __init__(self, adj=0, dist=0, weights=0):
        """
        adj: adjacency mat
        cost_func: a string that specifies cost function.
        supported: linear, affine, BRP, MM1
        """

        # weighted adj matrix
        self.adj = np.array(adj)

        self.adj = np.where(self.adj != 0, 1, 0)  # adjacency matrix

        self.weights = np.array(weights, dtype=np.float)
        self.adj_weights = np.array(weights, dtype=np.float) * self.adj

        # distance matrix
        self.dist = np.array(dist, dtype=np.float)
        self.adj_dist = self.dist * self.adj

        # wiring cost
        self.wiring_cost = self.adj_dist * self.adj_weights

        # reciprocal of weights
        self.rpl_weights = np.zeros(self.adj.shape)
        self.rpl_weights[self.weights != 0] = 1 / self.weights[self.weights != 0]

        # distance weight ratio
        self.dist_weight_ratio = self.adj_dist * self.rpl_weights

        # initialize arrays to store flows and costs
        # store all paths in a 3D list
        self.allpaths = [
            [[] for _ in range(self.adj.shape[1])] for _ in range(self.adj.shape[0])
        ]

    def dijkstra(self, s, t):
        """
        :param s: source
        :param t: target
        :return: shortest distant
        """
        q, seen = [(0, s)], set()

        while q:
            (distance, v) = heappop(q)
            if v not in seen:
                seen.add(v)
                if v == t:
                    return distance

                for u in np.nonzero(self.adj[v])[0]:
                    if u not in seen:
                        heappush(q, (distance + 1, u))

        return -1

    def dijkstra_weighted(self, s, t):
        """find shortest path between s and t (weighted network)"""
        q, seen = [(0, s)], set()

        while q:
            (distance, v) = heappop(q)
            if v not in seen:
                seen.add(v)
                if v == t:
                    return distance

                for u in np.nonzero(self.adj[v])[0]:
                    if u not in seen:
                        heappush(q, (distance + self.dist_weight_ratio[v, u], u))

        return -1

    def _findallpath_recursive(self, v, u, visited, path, allpaths, cutoff):

        visited[v] = True
        path.append(v)

        if len(path) < cutoff:
            if v == u:  # if the current vertice is the destination
                allpaths.append(path[:])  # creat deep copy of path

            elif v < self.adj.shape[0]:  # if not
                for k in np.nonzero(self.adj[v])[0]:  # traverse the vertices next to v
                    if not visited[k]:
                        self._findallpath_recursive(
                            k, u, visited, path, allpaths, cutoff
                        )

        elif v == u:  # len(path == cutoff)
            allpaths.append(path[:])

        # mark u as unvisited
        visited[v] = False
        path.pop()

    def findallpaths(self, s, t, cutoff=None):
        """
        find all possible paths shorter than cutoff from source s to target t
        :param s: source node
        :param t: target node
        :param cutoff: a scalar value
        :return: a list of the paths from s to t
        """

        if cutoff is None:
            cutoff = self.dijkstra(s, t) + 1 + 1
            if cutoff == 1:  # i.e. path length = 0
                return False

        allpaths = []
        num_vertices = np.max(self.adj.shape)

        # all vertices are unvisited at the beginning
        visited = [False] * num_vertices
        path = []  # temp path

        self._findallpath_recursive(s, t, visited, path, allpaths, cutoff)
        self.allpaths[s][t] = allpaths
        return allpaths

    def construct_path_arrays(self, s, t):

        # list of matrix to store path flows
        path_arrays = np.empty((0, self.adj.shape[0], self.adj.shape[1]))

        for path in self.allpaths[s][t]:
            path_array_tmp = np.zeros(self.adj.shape)
            # x index of the adj matrix
            index_x = [path[k] for k in range(len(path) - 1)]
            # y index of the adj matrix
            index_y = [path[k] for k in range(1, len(path))]
            path_array_tmp[index_x, index_y] = 1
            path_arrays = np.append(
                path_arrays, path_array_tmp[np.newaxis, :], axis=0
            )

        return path_arrays
