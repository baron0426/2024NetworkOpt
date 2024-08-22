import numpy as np


def get_mutual_information(cluster1, cluster2):
    N = len(cluster1)  # number of elements
    assert N == len(cluster2)
    shape = (max(cluster1) + 1, max(cluster2) + 1)
    L = np.zeros(shape)  # confusion matrix
    for (a, b) in zip(cluster1, cluster2):
        L[a, b] = L[a, b] + 1
    N1 = np.sum(L, axis=1)
    N2 = np.sum(L, axis=0)
    MI = L * np.log2(N * (np.diag(1/N1) @ L @ np.diag(1/N2)))
    HN1 = np.nansum(N1 * np.log2(N1/N))
    HN2 = np.nansum(N2 * np.log2(N2/N))
    NMI = (-2)*np.nansum(MI)/(HN1+HN2)
    return [MI/N, NMI]


if __name__ == '__main__':
    a = [0, 0, 0, 0, 0, 1,1,1,1,1]
    b = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    [L, MI] = get_mutual_information(a, b)
    print(L)
    print(MI)
    [L, MI] = get_mutual_information(b, b)
    print(L)
    print(MI)
