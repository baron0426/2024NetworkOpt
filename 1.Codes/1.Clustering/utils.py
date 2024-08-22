import numpy as np


def load_dynamic_matrix(file_list, normalized=True):
    dyn_mat = []
    for file in file_list:
        mat = np.loadtxt(file, delimiter=',')
        mat = (mat + mat.T) / 2
        if normalized:
            mat = mat / np.max(mat)
        dyn_mat.append(mat)
    dyn_mat = np.stack(dyn_mat, axis=0)
    return dyn_mat


def gen_modularity_densities(dyn_mat, cluster_results, typ=1):
    if dyn_mat.ndim != 3:
        raise Exception("bad matrix!")
    if cluster_results.ndim != 2:
        raise Exception("bad matrix!")
    modularity_densities = []
    for k in range(dyn_mat.shape[0]):
        if typ == 1:
            modularity_densities.append(gen_modularity_density(dyn_mat[k], cluster_results[k]))
        else:
            modularity_densities.append(gen_modularity_density2(dyn_mat[k], cluster_results[k]))
    return modularity_densities


def gen_modularity_density2(adj_mat, cluster_result):
    if adj_mat.ndim != 2:
        raise Exception("bad matrix!")
    (N, M) = adj_mat.shape
    assert N == M
    clusters = {}
    for i in range(N):
        if cluster_result[i] in clusters:
            clusters[cluster_result[i]]['size'] = clusters[cluster_result[i]]['size'] + 1
        else:
            clusters[cluster_result[i]] = {'size': 1, 'sum': 0}
        for j in range(i + 1, N):
            if cluster_result[i] == cluster_result[j]:
                clusters[cluster_result[i]]['sum'] = clusters[cluster_result[i]]['sum'] + adj_mat[i][j]
            else:
                clusters[cluster_result[i]]['sum'] = clusters[cluster_result[i]]['sum'] - adj_mat[i][j]
                if cluster_result[j] in clusters:
                    clusters[cluster_result[j]]['sum'] = clusters[cluster_result[j]]['sum'] - adj_mat[i][j]
                else:
                    clusters[cluster_result[j]] = {'size': 0, 'sum': -adj_mat[i][j]}
    retVal = 0
    for (_, cluster) in clusters.items():
        retVal = retVal + (cluster['sum'] / cluster['size'])
    return retVal


def gen_modularity_density(adj_mat, cluster_result):
    if adj_mat.ndim != 2:
        raise Exception("bad matrix!")
    clusters = {}
    for (index, val) in enumerate(cluster_result):
        if val not in clusters:
            clusters[val] = []
        clusters[val].append(index)
    # print(clusters)
    modularity_density = 0
    for cn in clusters:
        if len(clusters[cn]) == 1:
            continue
        l_in = 0
        l_out = 0
        for n in clusters[cn]:
            for m in range(adj_mat.shape[1]):
                if m == n:
                    continue
                if m in clusters[cn]:
                    l_in = l_in + adj_mat[n, m]
                else:
                    l_out = l_out + adj_mat[n, m]
        if l_in == 0:
            continue
        # print("Cluster: " + str(cn) + ", L_in: " + str(l_in) + ", L_out: " + str(l_out))
        modularity_density = modularity_density + (2 * l_out / (l_in))
    modularity_density = modularity_density / len(clusters)
    return modularity_density


def entropy_log2(x):
    return np.where(x != 0, np.log2(x), 0)


def get_mutual_information(cluster1, cluster2):
    N = len(cluster1)  # number of elements
    cluster1 = [int(x) for x in cluster1]
    cluster2 = [int(x) for x in cluster2]
    assert N == len(cluster2)
    shape = (max(cluster1) + 1, max(cluster2) + 1)
    L = np.zeros(shape)  # confusion matrix
    for (a, b) in zip(cluster1, cluster2):
        L[a, b] = L[a, b] + 1
    # print(L)
    N1 = np.sum(L, axis=1)
    N2 = np.sum(L, axis=0)
    # print(N1)
    # print(N2)
    MI = L * entropy_log2(N * (np.diag(1 / N1) @ L @ np.diag(1 / N2)))
    HN1 = np.nansum(N1 * entropy_log2(N1 / N))
    HN2 = np.nansum(N2 * entropy_log2(N2 / N))
    NMI = (-2) * np.nansum(MI) / (HN1 + HN2)
    return [MI / N, NMI]


def cluster_remap(cluster_result):
    (T, N) = cluster_result.shape
    new_cluster_result = np.zeros((T, N))
    new_cluster_result[0] = cluster_result[0]
    for t in range(1, T):
        [MI, _] = get_mutual_information(new_cluster_result[t - 1], cluster_result[t])
        index_descending = np.column_stack(np.unravel_index(np.argsort(MI.flatten())[::-1], MI.shape))
        clusters_prev = set(new_cluster_result[t - 1])
        clusters_now = set(cluster_result[t])
        remapping_result = []
        for [a, b] in index_descending:
            if b not in clusters_now:
                continue
            if a not in clusters_prev:
                continue
            remapping_result.append((a, b))
            clusters_prev.remove(a)
            clusters_now.remove(b)
        used_labels = set([u for (u, _) in remapping_result])
        for b in clusters_now:
            a = 0
            while a in used_labels:
                a = a + 1
            remapping_result.append((a, b))
            used_labels.add(a)
        remapping_result = {v: u for (u, v) in remapping_result}
        new_cluster_result[t] = [remapping_result[x] for x in cluster_result[t]]
    return new_cluster_result
