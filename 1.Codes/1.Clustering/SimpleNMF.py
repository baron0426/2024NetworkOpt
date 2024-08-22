from utils import *
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
def gen_cluster_result(B):
    if B.ndim != 3:
        raise Exception("bad matrix!")
    Z = np.zeros((B.shape[0], B.shape[1]))
    for t in range(B.shape[0]):
        Z[t] = np.argmax(B[t], axis=1)
    return Z


def gen_error(dyn_mat, B, F):
    err_list = []
    if B.ndim != 3:
        raise Exception("bad matrix!")
    if F.ndim != 3:
        raise Exception("bad matrix!")
    if dyn_mat.ndim != 3:
        raise Exception("bad matrix!")
    for k in range(dyn_mat.shape[0]):
        result = B[k] @ F[k]
        err = np.linalg.norm(dyn_mat[k] - result) / np.linalg.norm(dyn_mat[k])
        err_list.append(err)
    return err_list


if __name__ == '__main__':
    sample_files = ['./dataset/real/geant/preprocessed_daily/20050608.csv',
                    './dataset/real/geant/preprocessed_daily/20050609.csv',
                    './dataset/real/geant/preprocessed_daily/20050610.csv',
                    './dataset/real/geant/preprocessed_daily/20050611.csv',
                    './dataset/real/geant/preprocessed_daily/20050612.csv',
                    './dataset/real/geant/preprocessed_daily/20050613.csv',
                    './dataset/real/geant/preprocessed_daily/20050614.csv',
                    './dataset/real/geant/preprocessed_daily/20050615.csv',
                    './dataset/real/geant/preprocessed_daily/20050616.csv',
                    './dataset/real/geant/preprocessed_daily/20050617.csv',
                    './dataset/real/geant/preprocessed_daily/20050618.csv',
                    './dataset/real/geant/preprocessed_daily/20050619.csv',
                    './dataset/real/geant/preprocessed_daily/20050620.csv',
                    './dataset/real/geant/preprocessed_daily/20050621.csv',
                    './dataset/real/geant/preprocessed_daily/20050622.csv']

    # data_path = './dataset/synthetic/SYNFIXZ6/processed/'
    # sample_files = [data_path + str(k) + '.adjmat' for k in range(10)]
    dyn_mat = load_dynamic_matrix(sample_files)
    T = dyn_mat.shape[0]
    n = dyn_mat.shape[1]
    k = 4
    B = np.zeros((T, n, k))
    F = np.zeros((T, k, n))
    for t in range(T):
        model = NMF(n_components=k, solver='mu', init='random')
        B[t] = model.fit_transform(dyn_mat[t])
        F[t] = model.components_
    err = gen_error(dyn_mat, B, F)
    print(err)
    plt.plot(err)
    plt.show()
    Z = gen_cluster_result(B)
    mod = gen_modularity_densities(dyn_mat, Z)
    mod = [x if x < 5 else 5 for x in mod]
    print(mod)
    plt.plot(mod)
    plt.show()
    nmi = []
    for k in range(1, len(Z)):
        [_, tmp] = get_mutual_information(Z[k - 1], Z[k])
        nmi.append(tmp)
    plt.plot(nmi)
    plt.show()
    # Z_standard = np.zeros(Z.shape)
    # for k in range(len(Z)):
    #     tmp = np.loadtxt('./dataset/synthetic/SYNFIXZ6/' + str(k) + '.comm')
    #     Z_standard[k] = tmp[:, 1]
    # nmi_standard = []
    # for k in range(len(Z)):
    #     [_, tmp] = get_mutual_information(Z_standard[k], Z[k])
    #     nmi_standard.append(tmp)
    # plt.plot(nmi_standard)
    # plt.show()
    # print(nmi_standard)
