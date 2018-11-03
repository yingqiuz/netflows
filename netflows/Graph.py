# coding: utf-8

from __future__ import print_function
import numpy as np


# In[52]:

class Graph:
    def __init__(self, adj=0, dist=0, weights=0):
        """
        adj: adjacency mat
        cost_func: a string that specifies cost function. support: linear, affine, BRP, MM1
        """
        # weighted adj matrix
        self.adj = np.array(adj).T
        self.adj = np.where(self.adj != 0, 1, 0) # binary matrix
        self.adj_weights = np.array(weights, dtype=np.float).T # weighted matrix
        # distance matrix
        self.dist = np.array(dist, dtype=np.float).T
        self.adj_dist = self.dist * self.adj

        self.weights = np.array(weights, dtype=np.float).T

        # wiring cost
        self.wiring_cost = np.zero(self.adj.shape)
        self.wiring_cost[self.adj==1] = self.dist[self.adj==1] * self.weights[self.adj==1]
        # distance weight ratio
        self.weights[self.weights == 0] = np.inf
        self.dist_weight_ratio = self.dist / self.weights
        self.dist_weight_ratio[self.adj == 0 ] = 0
        self.adj_dist = np.array(dist, dtype=np.float) * self.adj
        #self.dist_weight_ratio = np.copy(self.adj_dist)
        self.allpaths = [[[] for k in range(self.adj.shape[1])] for kk in range(self.adj.shape[0])] # store all paths in a 3D list
        self.WEflowsLinear, self.WEflowsAffine, self.WEflowsBPR = [ [[[] for k in range(self.adj.shape[1])] for kk in range(self.adj.shape[0])] ] * 3
        self.WEflowsLinear_edge, self.WEflowsAffine_edge, self.WEflowsBPR_edge = [ [[[] for k in range(self.adj.shape[1])] for kk in range(self.adj.shape[0])] ] * 3        
        self.SOflowsLinear, self.SOflowsAffine, self.SOflowsBPR = [ [[[] for k in range(self.adj.shape[1])] for kk in range(self.adj.shape[0])] ] * 3
        self.SOflowsLinear_edge, self.SOflowsAffine_edge, self.SOflowsBPR_edge = [ [[[] for k in range(self.adj.shape[1])] for kk in range(self.adj.shape[0])] ] * 3        
        self.WEcostsLinear, self.WEcostsAffine, self.WEcostsBPR = [ [[[] for k in range(self.adj.shape[1])] for kk in range(self.adj.shape[0])] ] * 3
        self.SOcostsLinear, self.SOcostsAffine, self.SOcostsBPR = [ [[[] for k in range(self.adj.shape[1])] for kk in range(self.adj.shape[0])] ] * 3
        #self.cost_func_string = cost_func_string
        
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
            cutoff = np.max(self.adj.shape)
        allpaths = []
        num_vertices = np.max(self.adj.shape)    
        visited = [False] * (num_vertices) # all vertices are unvisited at the beginning 
        path = [] # temp path
    
        self._findallpath_recursive(s, t, visited, path, allpaths, cutoff)
        self.allpaths[s][t] = allpaths
        return allpaths
    
