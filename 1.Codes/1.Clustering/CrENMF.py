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
    C = np.zeros(k)
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
        A[j, :] = u
        B[j, :] = v
        C[j] = np.abs(sigma * S[j])
    return [A.T, np.diag(C), B]

def CrENMF2(dyn_mat, k_min, k_max, beta, gamma, iters=100):
    if dyn_mat.ndim != 3:
        raise Exception("bad matrix!")
    T = dyn_mat.shape[0]
    n = dyn_mat.shape[1]
    B = [None] * T
    H = [None] * T
    F = [None] * T
    k_list = [0] * T
    cluster_result = np.zeros((T, n))
    for t in range(T):
        print("Processing: t={}".format(t))
        if t > 0:
            Xe = B[t-1] / np.sqrt(np.sum((B[t-1])**2, axis=1)[:, np.newaxis])
            Xe = Xe @ Xe.T
            Xe = Xe - np.diag(np.diag(Xe))
            De = np.diag(np.sum(Xe, axis=1))
        max_mod = None
        for k in range(k_min, k_max + 1):
            print("Processing: k={}".format(k))
            [tempB, tempH, tempF] = svd_initialization(dyn_mat[t], k)
            for _ in range(iters):
                prevB = tempB
                prevH = tempH
                prevF = tempF
                tempH = ((prevB.T @ dyn_mat[t] @ prevF.T) / ((prevB.T @ prevB @ prevH @ prevF @ prevF.T)+np.finfo(np.float64).eps)) * prevH
                if t == 0:
                    tempB = ((dyn_mat[t] @ prevF.T @ prevH.T) / ((prevB @ prevH @ prevF @ prevF.T @prevH.T)+np.finfo(np.float64).eps)) * prevB
                    tempF = ((prevH.T @ prevB.T @ dyn_mat[t])/((prevH.T @ prevB.T @ prevB @ prevH @ prevF)+np.finfo(np.float64).eps)) * prevF
                else:
                    tempB = (((dyn_mat[t] @ prevF.T @ prevH.T) + (beta * Xe @ prevB)) / ((prevB @ prevH @ prevF @ prevF.T @ prevH.T) + (beta * De @ prevB) + np.finfo(np.float64).eps)) * prevB
                    tempF = (((prevH.T @ prevB.T @ dyn_mat[t]) + (gamma * prevF @ Xe)) / ((prevH.T @ prevB.T @ prevB @ prevH @ prevF) + (gamma * prevF @ De) + np.finfo(np.float64).eps)) * prevF
                tempB[tempB < 0] = 0  # according to KKT condition, either bij=0 or partial derivitive = 0
                tempH[tempH < 0] = 0  # according to KKT condition, either hij=0 or partial derivitive = 0
                tempF[tempF < 0] = 0  # according to KKT condition, either fij=0 or partial derivitive = 0
                tempB = tempB / np.sum(tempB, axis=0)[np.newaxis, :]
                tempF = tempF / np.sum(tempF, axis=1)[:, np.newaxis]
            # Select best K and save to B, H, F
            if k == k_min:
                cluster_result[t] = np.argmax(tempB, axis=1)
                max_mod = gen_modularity_density2(dyn_mat[t], cluster_result[t])
                B[t] = tempB
                H[t] = tempH
                F[t] = tempF
                k_list[t] = k
            else:
                temp_result = np.argmax(tempB, axis=1)
                temp_mod = gen_modularity_density2(dyn_mat[t], temp_result)
                if temp_mod > max_mod:
                    max_mod = temp_mod
                    cluster_result[t] = temp_result
                    B[t] = tempB
                    H[t] = tempH
                    F[t] = tempF
                    k_list[t] = k
    return [B, H, F, k_list, cluster_result]


if __name__ == '__main__':
    data_path = './dataset/real/geant/preprocessed_daily/'
    sample_files = [(data_path + x) for x in os.listdir(data_path)]
    output_folder_base = './results/CrENMF2/'
    dyn_mat = load_dynamic_matrix(sample_files)
    hps = [0.3, 0.5, 0.7]
    k_min = 3
    k_max = 6
    for beta in hps:
        for gamma in hps:
            print("CrENMF for hyperparameter beta={}. gamma={}".format(beta, gamma))
            output_folder = output_folder_base + '{}_{}/'.format(int(10*beta), int(10*gamma))
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            [B, H, F, k_list, cluster_result] = CrENMF2(dyn_mat, k_min, k_max, beta, gamma)
            print("Done!")
            np.savetxt('{}/k_list.csv'.format(output_folder), k_list, fmt='%d', delimiter=',')
            np.savetxt('{}/cluster_result.csv'.format(output_folder), cluster_result, fmt='%d', delimiter=',')
            for t in range(len(B)):
                np.savetxt('{}/B_{}.csv'.format(output_folder, t), B[t], fmt='%f', delimiter=',')
                np.savetxt('{}/H_{}.csv'.format(output_folder, t), H[t], fmt='%f', delimiter=',')
                np.savetxt('{}/F_{}.csv'.format(output_folder, t), F[t], fmt='%f', delimiter=',')



