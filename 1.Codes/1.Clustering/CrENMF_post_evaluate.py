import numpy as np
import matplotlib.pyplot as plt
from utils import *

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

    # data_path = './dataset/synthetic/SYNFIXZ3/processed/'
    # sample_files = [data_path + str(k) + '.adjmat' for k in range(10)]
    dyn_mat = load_dynamic_matrix(sample_files)
    Z = np.loadtxt('./CrENMF/cluster_results_GEANT.csv', delimiter=',')
    Z = Z.T

    mod = gen_modularity_densities(dyn_mat, Z)
    mod = [x if x < 10 else 10 for x in mod]
    print(mod)
    plt.plot(mod)
    plt.show()
    nmi = []
    for k in range(1, len(Z)):
        [_, tmp] = get_mutual_information(Z[k - 1], Z[k])
        nmi.append(tmp)
    plt.plot(nmi)
    plt.show()

    Z_standard = np.zeros(Z.shape)
    # for k in range(len(Z)):
    #     tmp = np.loadtxt('./dataset/synthetic/SYNFIXZ3/' + str(k) + '.comm')
    #     Z_standard[k] = tmp[:, 1]
    # nmi_standard = []
    # for k in range(len(Z)):
    #     [_, tmp] = get_mutual_information(Z_standard[k], Z[k])
    #     nmi_standard.append(tmp)
    # plt.plot(nmi_standard)
    # plt.show()
    # print(nmi_standard)
