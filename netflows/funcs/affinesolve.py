import numpy as np
import scipy

from netflows.funcs.costfuncs import affine_cost,affine_SO_obj,affine_WE_obj

def WE(G, s, t, tol = 1e-12, maximum_iter = 10000, cutoff = None, a = None, a0 = None):
    """
    single pair Wardrop Equilibrium flow
    s: source
    t: destination
    tol: tolerance
    gamma: descent speed
    """
    if cutoff = None:
        print("Warning: cutoff not specified. it may take hugh memory to find all paths")
        cutoff = min(G.adj.shape)

    allpaths = G.allpaths[s][t]
    if allpaths == []:
        allpaths = G.findallpaths(s, t, cutoff)

    if a == None:
        a = G.adj_dist

    if a0 == None:
        a0 = G.adj_weights

    return _WE(G, s, t, tol, maximum_iter, allpaths, a, a0)

def SO(G, s, t, tol=1e-12, maximum_iter = 10000, cutoff = None, a = None, a0 = None):
    """
    :param G:
    :param s:
    :param t:
    :param tol:
    :param maximum_iter:
    :param cutoff:
    :return:
    """
    if cutoff = None:
        print("Warning: cutoff not specified. it may take hugh memory to find all paths")
        cutoff = min(G.adj.shape)

    allpaths = G.allpaths[s][t]
    if allpaths = []:
        allpaths = G.findallpaths(s, t, cutoff)

    if a == None:
        a = G.adj_dist

    if a0 == None:
        a0 = G.adj_weights

    return _SO(G, s, t, tol, maximum_iter, allpaths, a, a0)

def _WE(G, s, t, tol, maximum_iter, allpaths, a, a0):
    """
    :param G:
    :param s:
    :param t:
    :param tol:
    :param maximum_iter:
    :param allpaths:
    :return:
    """
    #a0 = G.adj_weights

    num_variables = len(allpaths)  # the number of paths from s to t

    # x is the flow vector (path formulation)
    x = np.ones((num_variables,)) / num_variables  # initial value
    # find equilibrium -- convex optimization
    # map to matrix
    path_arrays = np.empty((0, G.adj.shape[0], G.adj.shape[1]))  # list of matrix to store path flows

    for path in allpaths:
        path_array_tmp = np.zeros(G.adj.shape)
        index_x = [path[k] for k in range(len(path) - 1)]  # x index of the adj matrix
        index_y = [path[k] for k in range(1, len(path))]  # y index of the adj matrix
        path_array_tmp[index_x, index_y] = 1
        path_arrays = np.append(path_arrays, path_array_tmp[np.newaxis, :], axis=0)

    # element (i, j) is the total flow on edge (i,j)
    allflows = np.sum(path_arrays * x.reshape(num_variables, 1, 1), axis=0)

    obj_fun = np.sum(affine_WE_obj(allflows, a, a0_mat=a0))
    # obj_fun = np.sum(self.WE_obj(allflows), axis = None)
    total_cost = np.sum(allflows * affine_cost(allflows, a, a0=a0))

    #total_traveltime = np.sum(cost_funcs.affine_cost(allflows, a, a0=a0)) --- useless
    print('initial cost %f' % total_cost)
    print('initial flow', x)
    print('------solve the Wardrop Equilibrium------')

    gradients = np.array(
        [np.sum(affine_cost(allflows, a, a0=a0) * (
                    path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1, 0) *
                    path_arrays[-1]))
         for k in range(num_variables - 1)]
    )

    for k in range(maximum_iter):  # maximal iteration 10000

        prev_obj_fun = np.copy(obj_fun)
        prev_x = np.copy(x)
        #prev_allflows = np.copy(allflows)
        #prev_total_cost = np.copy(total_cost)
        #prev_total_traveltime = np.copy(total_traveltime)
        #prev_gradients = np.copy(gradients)
        result = scipy.optimize.linprog(gradients, bounds=(0, 1),
                                        options={'maxiter': 1000, 'disp': False, 'tol': 1e-12, 'bland': True})

        # step size determination
        gamma = 2 / (k + 1 + 2)
        # update x
        x[:-1] = prev_x[:-1] + gamma * (result.x - x[:-1])
        x[-1] = 1 - np.sum(x[:-1])  # the flow in the last path

        #if np.sum(np.where(x < 0, 1, 0)) > 0:  # flow in at least one path is negtive
        #    print('Iteration %d: The total cost is %f, the total travel time is %f, and the flow is ' % (
        #    k, prev_total_cost, prev_total_traveltime), prev_x)
        #    self.WEflowsAffine[s][t] = prev_x
        #    self.WEcostsAffine[s][t] = prev_total_cost
        #    self.WEflowsAffine_edge[s][t] = prev_allflows
        #    return prev_total_cost, prev_total_traveltime, prev_x

        allflows = np.sum(path_arrays * x.reshape(num_variables, 1, 1), axis = 0)
        obj_fun = np.sum(affine_WE_obj(allflows, a, a0_mat=a0), axis=None)
        diff_value = obj_fun - prev_obj_fun
        total_cost = np.sum(allflows * affine_cost(allflows, a, a0=a0), axis=None)
        #total_traveltime = np.sum(cost_funcs.affine_cost(allflows, self.adj_dist, a0=a0), axis=None)
        if np.abs(diff_value / prev_obj_fun) < tol:
            print('Wardrop equilibrium found. total cost %d' % total_cost)
            print('flows (path formulation) are', x)
            G.WEflowsAffine[s][t] = x
            G.WEcostsAffine[s][t] = total_cost
            G.WEflowsAffine_edge[s][t] = allflows
            return total_cost,  x
        #print('Iteration %d: The total cost is %f, the total travel time is %f, and the flow is ' % (
        #k, total_cost, total_traveltime), x)

        # new gradients
        gradients = np.array(
            [np.sum(affine_cost(allflows, G.adj_dist, a0=a0) * (
                        path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1,
                                                                                         0) * path_arrays[-1]))
             for k in range(num_variables - 1)]
        )
        #gamma = np.inner(x[:-1] - prev_x[:-1], gradients - prev_gradients) / np.inner(gradients - prev_gradients,
                                                                                      #gradients - prev_gradients)


    print('global minimum not found')
    return

def _SO(G, s, t, tol, maximum_iter, allpaths, a, a0):
    """
    single pair System Optimal flow, affine cost function
    s: source
    t: destination
    tol: tolerance
    gamma: descent speed
    """

    num_variables = len(allpaths)  # the number of paths from s to t

    # x is the flow vector (path formulation)
    x = np.ones((num_variables,)) / num_variables  # initial value
    # find equilibrium -- convex optimization
    # map to matrix
    path_arrays = np.empty((0, G.adj.shape[0], G.adj.shape[1]))  # list of matrix to store path flows

    for path in allpaths:
        path_array_tmp = np.zeros(G.adj.shape)
        index_x = [path[k] for k in range(len(path) - 1)]  # x index of the adj matrix
        index_y = [path[k] for k in range(1, len(path))]  # y index of the adj matrix
        path_array_tmp[index_x, index_y] = 1
        path_arrays = np.append(path_arrays, path_array_tmp[np.newaxis, :], axis=0)

    # element (i, j) is the total flow on edge (i,j)
    allflows = np.sum(path_arrays * x.reshape(num_variables, 1, 1), axis=0)

    obj_fun = np.sum(affine_SO_obj(allflows, a, a0_mat=a0)) # obj_fun is the total cost
    # total_cost = np.sum(allflows * cost_funcs.linear_cost(allflows, self.adj_dist))
    # total_traveltime = np.sum(cost_funcs.affine_cost(allflows, self.adj_dist, a0=a0))
    print('initial cost %f' % (obj_fun))
    print('initial flow', x)
    print('------solve the system optimal flow------')

    # gradients
    gradients = np.array(
        [np.sum(affine_cost(allflows, a, a0=a0) * (
                    path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1, 0) *
                    path_arrays[-1]))
         for k in range(num_variables - 1)]
    ) + np.array(
        [np.sum(allflows * a * (
                    path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1, 0) *
                    path_arrays[-1]))
         for k in range(num_variables - 1)]
    )

    for k in range(maximum_iter):  # maximal iteration 10000

        prev_obj_fun = np.copy(obj_fun)
        prev_x = np.copy(x)
        #prev_allflows = np.copy(allflows)
        #prev_total_traveltime = np.copy(total_traveltime)
        #prev_gradients = np.copy(gradients)

        # determine step size
        result = scipy.optimize.linprog(gradients, bounds=(0, 1),
                                        options={'maxiter': 1000, 'disp': False, 'tol': 1e-12, 'bland': True})

        # step size determination
        gamma = 2 / (k + 1 + 2)
        # update x
        x[:-1] = prev_x[:-1] + gamma * (result.x - x[:-1])
        x[-1] = 1 - np.sum(x[:-1])  # the flow in the last path

        #if np.sum(np.where(x < 0, 1, 0)) > 0:  # flow in at least one path is negtive
        #    print('Iteration %d: The total cost is %f, and total travel time is %f, and the flow is ' % (
        #    k, prev_obj_fun, prev_total_traveltime), prev_x)
        #    self.SOflowsAffine[s][t] = prev_x
        #    self.SOcostsAffine[s][t] = prev_obj_fun
        #    self.SOflowsAffine_edge[s][t] = prev_allflows
        #    return prev_obj_fun, prev_total_traveltime, prev_x

        allflows = np.sum(path_arrays * x.reshape(num_variables, 1, 1), axis=0)
        obj_fun = np.sum(affine_SO_obj(allflows, a, a0_mat=a0), axis=None)
        diff_value = obj_fun - prev_obj_fun
        #total_traveltime = np.sum(cost_funcs.affine_cost(allflows, self.adj_dist, a0=a0), axis=None)

        if np.abs(diff_value / prev_obj_fun) < tol:
            print('system optimum found: total cost %f' % obj_fun)
            print('the flows are (path formulation)', x)
            G.SOflowsAffine[s][t] = x
            G.SOcostsAffine[s][t] = obj_fun
            G.SOflowsAffine_edge[s][t] = allflows
            return obj_fun, x

        #print('Iteration %d: The total cost is %f, the total travel time is %f, and the flow is ' % (
        #k, obj_fun, total_traveltime), x)

        # update gradients
        gradients = np.array(
            [np.sum(affine_cost(allflows, a, a0=a0) * (
                        path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1,
                                                                                         0) * path_arrays[-1]))
             for k in range(num_variables - 1)]
        ) + np.array(
            [np.sum(allflows * a * (
                        path_arrays[k] * np.where(path_arrays[-1] == 0, 1, 0) - np.where(path_arrays[k] == 0, 1,
                                                                                         0) * path_arrays[-1]))
             for k in range(num_variables - 1)]
        )
        #gamma = np.inner(x[:-1] - prev_x[:-1], gradients - prev_gradients) / np.inner(gradients - prev_gradients,
                                                                                      #gradients - prev_gradients)

    print('global minimum not found')
    return