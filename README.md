# netflows

This package finds optimal traffic assignment on networks to minimize total travel cost of all users, or estimates excessive travel cost due to lack of coordination if users choose their routes selfishly.

[![Build Status](https://travis-ci.com/yingqiuz/netflows.svg?token=GCAiuUe1sWERsysgW6zt&branch=master)](https://travis-ci.com/yingqiuz/netflows)

## Overview

## Install
Make sure you have Python>=3.5 installed. To install this package, open a terminal and type the following:

```shell
git clone https://github.com/yingqiuz/netflows.git
cd netflows
python setup.py install
```

## Usage

### find all possible paths
To find all paths that are shorter than k steps (i.e., binary distance) from a source node *s* to a target node *t*, run the following:
```python
from netflows import Graph

# create the Graph object
G = Graph(adj=adjacency_matrix, dist=distance_matrix, weights=weight_matrix)
G.findallpaths(s, t, cutoff=k)
``` 
which returns a list of all the possible paths from *s* to *k* shorter than *k*. Each possible path is represented by a list storing the index of the nodes. **NB**: in the adjacency/distance/weight matrix, element *(i, j)* denotes the directionality from *i* to *j*.

# find shortest paths
To find the shortest path from *s* to *t*:
```python
G.dijkstra(s, t)
```
which returns the binary shortest distance from *s* to *t*, or the weighted version:
```python
G.dijkstra_weighted(s, t)
```

### System Optimal flow
To find the system optimal flow assignment that minimizes the total travel cost of all the network users:
```python
from netflows import system_optimal_linear_solve
# find the optimal flow assignment and the total travel cost
flows_path_formulation, flows_edge_formulation, total_travel_cost, edge_travel_cost = system_optimal_linear_solve(G, s, t, tol=1e-8, maximum_iter=100000, cutoff=k)
```

### Wardrop Equilibrium flow