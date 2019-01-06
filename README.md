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
allpaths = G.findallpaths(s, t, cutoff=k)
``` 
`allpaths`is  a list of all the possible paths from *s* to *k* shorter than *k*. Each element is a list storing the nodes' indices in the order that the users traverse from *s* to *t*. **NB**: in the adjacency/distance/weight matrix, element *i, j* denotes the directionality from *i* to *j*.

### find shortest paths
To find the shortest path from *s* to *t*:
```python
d = G.dijkstra(s, t)
```
which returns the binary shortest distance from *s* to *t*, or the weighted version:
```python
d = G.dijkstra_weighted(s, t)
```

### System Optimal flow
To find the system optimal flow assignment that minimizes the total travel cost of all the network users:
```python
from netflows import system_optimal_linear_solve

# find the optimal flow assignment and the total travel cost
flows_path_formulation, flows_edge_formulation, total_travel_cost, edge_travel_cost = system_optimal_linear_solve(G, s, t, tol=1e-8, maximum_iter=100000, cutoff=k)
```
`flows_path_formulation` is a list of flows corresponding to the elements in `allpaths`. `flows_edge_formulation` is a matrix with element *i, j* storing the flow on edge *(i, j)*. `total_travel_cost` is the total travel cost incurred by all the users (a scalar value), and `edge_travel_cost`is a matrix with element *i, j* storing the travel cost on edge *(i, j)* incurred by the users that traverse this edge.

Also supported: `system_optimal_affine_solve`, `system_optimal_bpr_solve`.

### Wardrop Equilibrium flow
To estimate the travel cost due to lack of coordination (i.e., to find the Wardrop Equilibrium flow):
```python
from netflows import wardrop_equilibrium_linear_solve

# find the optimal flow assignment and the total travel cost
flows_path_formulation, flows_edge_formulation, total_travel_cost, edge_travel_cost = wardrop_equilibrium_linear_solve(G, s, t, tol=1e-8, maximum_iter=100000, cutoff=k)
```
Likewise, `flows_path_formulation` is a list of flows corresponding to the elements in `allpaths`. `flows_edge_formulation` is a matrix with element *i, j* storing the flow on edge *(i, j)*. `total_travel_cost` is the total travel cost incurred by all the users (a scalar value), and `edge_travel_cost`is a matrix with element *i, j* storing the travel cost on edge *(i, j)* incurred by the users that traverse this edge.

Aso supported: `wardrop_equilibrium_affine_solve`, `wardrop_equilibrium_bpr_solve`.

