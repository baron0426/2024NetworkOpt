from utils import *
import os


def svd_initialization(mat, k):
    (n, m) = mat.shape
    [U, S, Vh] = np.linalg.svd(mat, compute_uv=True)
    S = [x for x in S if abs(x) > 1e-10]
    r = len(S)
    if k > r:
        k = r
    A = np.zeros((k, n))
    B = np.zeros((k, m))
    for j in range(k):
        x = U[:, j]
        y = Vh[j, :]
        xp = x * (x >= 0)
        xn = -x * (x <= 0)
        yp = y * (y >= 0)
        yn = -y * (y <= 0)
        xp_norm = np.linalg.norm(xp)
        yp_norm = np.linalg.norm(yp)
        xn_norm = np.linalg.norm(xn)
        yn_norm = np.linalg.norm(yn)
        mp = xp_norm * yp_norm
        mn = xn_norm * yn_norm
        if mp > mn:
            sigma = mp
            u = xp / xp_norm
            v = yp / yp_norm
        else:
            sigma = mn
            u = xn / xn_norm
            v = yn / yn_norm
        A[j, :] = np.sqrt(S[j] * sigma) * u
        B[j, :] = np.sqrt(S[j] * sigma) * v
    return [A.T, B]


def gen_cluster_result(B):
    if B.ndim != 3:
        raise Exception("bad matrix!")
    Z = np.zeros((B.shape[0], B.shape[1]))
    for t in range(B.shape[0]):
        Z[t] = np.argmax(B[t], axis=1)
    return Z


def GrENMF(dyn_mat, k, alpha, iters=1000):
    if dyn_mat.ndim != 3:
        raise Exception("bad matrix!")
    T = dyn_mat.shape[0]
    n = dyn_mat.shape[1]
    m = dyn_mat.shape[2]
    B = np.zeros((T, n, k))
    F = np.zeros((T, k, m))
    for t in range(T):
        if t > 0:
            last_l = np.diag(np.sum(dyn_mat[t - 1], axis=1)) - dyn_mat[t - 1]
        [B[t], F[t]] = svd_initialization(dyn_mat[t], k)
        for _ in range(iters):
            prevB = B[t]
            B[t] = ((dyn_mat[t] @ F[t].T) / (B[t] @ F[t] @ F[t].T)) * B[t]
            if t == 0:
                F[t] = ((prevB.T @ dyn_mat[t]) / (prevB.T @ prevB @ F[t])) * F[t]
            else:
                F[t] = ((prevB.T @ dyn_mat[t]) / ((prevB.T @ prevB @ F[t]) + (alpha * F[t] @ last_l))) * F[t]
            B[t][B[t] < 0] = 0  # according to KKT condition, either bij=0 or partial derivitive = 0
            F[t][F[t] < 0] = 0  # according to KKT condition, either fij=0 or partial derivitive = 0
            # B[t] = B[t] / np.sum(B[t], 1)[:, np.newaxis]
            # F[t] = F[t] / np.sum(F[t], 0)[np.newaxis, :]
    return [B, F]


def GrENMF2(dyn_mat, k_min, k_max, alpha, iters=100):
    if dyn_mat.ndim != 3:
        raise Exception("bad matrix!")
    T = dyn_mat.shape[0]
    n = dyn_mat.shape[1]
    m = dyn_mat.shape[2]
    B = [None] * T
    F = [None] * T
    k_list = [0] * T
    cluster_result = np.zeros((T, n))
    for t in range(T):
        if t > 0:
            last_l = np.diag(np.sum(dyn_mat[t - 1], axis=1)) - dyn_mat[t - 1]
        max_mod = None
        for k in range(k_min, k_max + 1):
            [tempB, tempF] = svd_initialization(dyn_mat[t], k)
            for _ in range(iters):
                prevB = tempB
                tempB = ((dyn_mat[t] @ tempF.T) / ((tempB @ tempF @ tempF.T)+np.finfo(np.float64).eps)) * tempB
                if t == 0:
                    tempF = ((prevB.T @ dyn_mat[t]) / ((prevB.T @ prevB @ tempF) + np.finfo(np.float64).eps)) * tempF
                else:
                    tempF = ((prevB.T @ dyn_mat[t]) / ((prevB.T @ prevB @ tempF) + (alpha * tempF @ last_l) + np.finfo(np.float64).eps)) * tempF
                tempB[tempB < 0] = 0  # according to KKT condition, either bij=0 or partial derivitive = 0
                tempF[tempF < 0] = 0  # according to KKT condition, either fij=0 or partial derivitive = 0
            if k == k_min:
                cluster_result[t] = np.argmax(tempB, axis=1)
                max_mod = gen_modularity_density2(dyn_mat[t], cluster_result[t])
                B[t] = tempB
                F[t] = tempF
                k_list[t] = k
            else:
                temp_result = np.argmax(tempB, axis=1)
                temp_mod = gen_modularity_density2(dyn_mat[t], temp_result)
                if temp_mod > max_mod:
                    max_mod = temp_mod
                    cluster_result[t] = temp_result
                    B[t] = tempB
                    F[t] = tempF
                    k_list[t] = k

    return [B, F, k_list, cluster_result]


def gen_error(dyn_mat, B, F):
    err_list = []
    # if B.ndim != 3:
    #     raise Exception("bad matrix!")
    # if F.ndim != 3:
    #     raise Exception("bad matrix!")
    if dyn_mat.ndim != 3:
        raise Exception("bad matrix!")
    for k in range(dyn_mat.shape[0]):
        result = B[k] @ F[k]
        err = np.linalg.norm(dyn_mat[k] - result) / np.linalg.norm(dyn_mat[k])
        err_list.append(err)
    return err_list


if __name__ == '__main__':
    data_path = './dataset/real/geant/preprocessed_daily/'
    sample_files = [(data_path + x) for x in os.listdir(data_path)]
    # sample_files = ['./dataset/real/geant/preprocessed_daily/20050608.csv',
    #                 './dataset/real/geant/preprocessed_daily/20050609.csv',
    #                 './dataset/real/geant/preprocessed_daily/20050610.csv',
    #                 './dataset/real/geant/preprocessed_daily/20050611.csv',
    #                 './dataset/real/geant/preprocessed_daily/20050612.csv',
    #                 './dataset/real/geant/preprocessed_daily/20050613.csv',
    #                 './dataset/real/geant/preprocessed_daily/20050614.csv',
    #                 './dataset/real/geant/preprocessed_daily/20050615.csv',
    #                 './dataset/real/geant/preprocessed_daily/20050616.csv',
    #                 './dataset/real/geant/preprocessed_daily/20050617.csv',
    #                 './dataset/real/geant/preprocessed_daily/20050618.csv',
    #                 './dataset/real/geant/preprocessed_daily/20050619.csv',
    #                 './dataset/real/geant/preprocessed_daily/20050620.csv',
    #                 './dataset/real/geant/preprocessed_daily/20050621.csv',
    #                 './dataset/real/geant/preprocessed_daily/20050622.csv']

    # data_path = './dataset/synthetic/SYNFIXZ6/processed/'
    # sample_files = [data_path + str(k) + '.adjmat' for k in range(10)]
    dyn_mat = load_dynamic_matrix(sample_files)
    alphas = [0.3, 0.5, 0.7]
    output_folder_base = './results/GrENMF2/'
    for alpha in alphas:
        print("GrENMF for hyperparameter alpha={}".format(alpha))
        output_folder = output_folder_base + '{}/'.format(int(10 * alpha))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        [B, F, k_list, cluster_result] = GrENMF2(dyn_mat, 3, 6, alpha)
        print("Done!")
        np.savetxt('{}/k_list.csv'.format(output_folder), k_list, fmt='%d', delimiter=',')
        np.savetxt('{}/cluster_result.csv'.format(output_folder), cluster_result, fmt='%d', delimiter=',')
        for t in range(len(B)):
            np.savetxt('{}/B_{}.csv'.format(output_folder, t), B[t], fmt='%f', delimiter=',')
            np.savetxt('{}/F_{}.csv'.format(output_folder, t), F[t], fmt='%f', delimiter=',')

    # err = gen_error(dyn_mat, B, F)
    # print(err)
    # plt.plot(err)
    # plt.show()
    # mod = gen_modularity_densities(dyn_mat, Z)
    # mod = [x if x < 10 else 10 for x in mod]
    # print(mod)
    # plt.plot(mod)
    # plt.show()
    # nmi = []
    # for k in range(1, len(Z)):
    #     [_, tmp] = get_mutual_information(Z[k - 1], Z[k])
    #     nmi.append(tmp)
    # plt.plot(nmi)
    # plt.show()
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
