import numpy as np
from numpy.linalg import norm
def get_reconstructed_adj(X=None, node_l=None):
    """Compute the adjacency matrix from the learned embedding
    Returns:
    A numpy array of size #nodes * #nodes containing the reconstructed adjacency matrix.
    """
    if X is not None:
        node_num = X.shape[0]
        X = X
    else:
        node_num = node_l
    adj_mtx_r = np.zeros((node_num, node_num))
    for v_i in range(node_num):
        for v_j in range(node_num):
            if v_i == v_j:
                continue
            adj_mtx_r[v_i, v_j] = get_edge_weight(X, v_i, v_j)
    return adj_mtx_r


def get_edge_weight(X, i, j):
    return (np.dot(X[i, :], X[j, :])/(norm(X[i, :]) * norm(X[j, :])))



def get_edge_weight_hope(X, i, j, d):
    return np.dot(X[i, :d // 2], X[j, d // 2:])

#def get_edge_weight_avg(X, i, j):
#pass  


def get_edge_list_from_adj_mtrx(adj, threshold=0.5, is_undirected=False, edge_pairs=None):
    result = []
    node_num = adj.shape[0]
    if edge_pairs:
        for (st, ed) in edge_pairs:
            if adj[st, ed] >= threshold:
                result.append((st, ed, adj[st, ed]))
    else:
        for i in range(node_num):
            for j in range(node_num):
                if j == i:
                    continue
                if is_undirected and i >= j:
                    continue
                if adj[i, j] > threshold:
                    result.append((i, j, adj[i, j]))
    return result



def evaluateGraphReconstruction(digraph, embedding, X, node_l = None, isundirected = False, isweighted = True):
    pass