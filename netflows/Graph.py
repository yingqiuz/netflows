# coding: utf-8

from __future__ import print_function
import numpy as np

from heapq import heappop, heappush


# In[52]:

class Graph:
    def __init__(self, adj=0, dist=0, weights=0):
        """
        adj: adjacency mat
        cost_func: a string that specifies cost function. support: linear, affine, BRP, MM1
        """
        # weighted adj matrix
        self.adj = np.array(adj).T

        self.adj = np.where(self.adj != 0, 1, 0) # adjacency matrix

        self.weights = np.array(weights, dtype=np.float).T
        self.adj_weights = np.array(weights, dtype=np.float).T * self.adj# weighted matrix * self

        # distance matrix
        self.dist = np.array(dist, dtype=np.float).T
        self.adj_dist = self.dist * self.adj
        # wiring cost
        # self.wiring_cost = np.zeros(self.adj.shape)
        self.wiring_cost = self.dist * self.adj_weights

        # wiring cost
        self.wiring_cost = self.adj_dist * self.adj_weights

        # reciprocal of weights
        self.rpl_weights = np.zeros(self.adj.shape)
        self.rpl_weights[self.weights!=0] = 1 / self.weights[self.weights!=0]

        # distance weight ratio
        self.dist_weight_ratio = self.adj_dist * self.rpl_weights

        # initialize arrays to store flows and costs

        self.allpaths = [[[] for k in range(self.adj.shape[1])] for kk in range(self.adj.shape[0])] # store all paths in a 3D list
        self.WEflowsLinear, self.WEflowsAffine, self.WEflowsBPR = [ [[[] for k in range(self.adj.shape[1])] for kk in range(self.adj.shape[0])] ] * 3
        self.WEflowsLinear_edge, self.WEflowsAffine_edge, self.WEflowsBPR_edge = [np.zeros(self.adj.shape)] * 3

        self.SOflowsLinear, self.SOflowsAffine, self.SOflowsBPR = [ [[[] for k in range(self.adj.shape[1])] for kk in range(self.adj.shape[0])] ] * 3
        self.SOflowsLinear_edge, self.SOflowsAffine_edge, self.SOflowsBPR_edge = [np.zeros(self.adj.shape)] * 3

        self.WEcostsLinear, self.WEcostsAffine, self.WEcostsBPR = np.zeros(self.adj.shape) #[ [[[] for k in range(self.adj.shape[1])] for kk in range(self.adj.shape[0])] ] * 3
        self.WEcostsLinear_edge, self.WEcostsAffine_edge, self.WEcostsBPR_edge = [np.zeros(self.adj.shape)] * 3

        self.SOcostsLinear, self.SOcostsAffine, self.SOcostsBPR = np.zeros(self.adj.shape) #[ [[[] for k in range(self.adj.shape[1])] for kk in range(self.adj.shape[0])] ] * 3
        self.SOcostsLinear_edge, self.SOcostsAffine_edge, self.SOcostsBPR_edge = [np.zeros(self.adj.shape)] * 3

        # how many single source-target pairs are calculated
        self.total_pair = 0

    def _dijkstra(self, s, t):
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



    def _findallpath_recursive(self, v, u, visited, path, allpaths, cutoff):
        """
        adj: adj matrix
        s: source
        t: destination
        visited: vertices that have been visited
        path: temporary path
        allpaths: all paths...
        """
        # mark u as visited
        visited[v] = True
        path.append(v)
        
        if len(path) < cutoff:
            if v == u:  # if the current vertice is the destination
                allpaths.append(path[:]) # creat deep copy of path
                #print(path)
            
            elif v < self.adj.shape[0]:  # if not
                for k in np.nonzero(self.adj[v])[0]: # traverse the vertices next to v
                    if visited[k] == False:
                        self._findallpath_recursive(k, u, visited, path, allpaths, cutoff)
                        
        elif v == u: # len(path == cutoff)
            allpaths.append(path[:])
            #print(path)
            
            
        # mark u as unvisited
        visited[v] = False
        path.pop()
    
    def findallpaths(self, s, t, cutoff = None):
        """
        find all possible paths from source s to destination t
        adj: adj matrix
        s: source
        t: destination
        """
        if cutoff == None:
            cutoff = self._dijkstra(s, t) + 1 + 1
            if cutoff == 1: # i.e. path length = 0
                return False
        allpaths = []
        num_vertices = np.max(self.adj.shape)    
        visited = [False] * (num_vertices) # all vertices are unvisited at the beginning 
        path = [] # temp path
    
        self._findallpath_recursive(s, t, visited, path, allpaths, cutoff)
        self.allpaths[s][t] = allpaths
        return allpaths

    def construct_path_arrays(self, s, t):

        path_arrays = np.empty((0, self.adj.shape[0], self.adj.shape[1]))  # list of matrix to store path flows

        for path in self.allpaths[s][t]:
            path_array_tmp = np.zeros(self.adj.shape)
            index_x = [path[k] for k in range(len(path) - 1)]  # x index of the adj matrix
            index_y = [path[k] for k in range(1, len(path))]  # y index of the adj matrix
            path_array_tmp[index_x, index_y] = 1
            path_arrays = np.append(path_arrays, path_array_tmp[np.newaxis, :], axis=0)

        return path_arrays


