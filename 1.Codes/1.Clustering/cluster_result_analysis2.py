from utils import *
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    data_path = './dataset/real/geant/preprocessed_daily/'
    sample_files = [(data_path + x) for x in os.listdir(data_path)]
    dyn_mat = load_dynamic_matrix(sample_files)

    results_folder_base = './results/'
    baseline_cluster_result_path = './results/Spectral/cluster_result.csv'
    baseline_cluster_result = np.loadtxt(baseline_cluster_result_path, delimiter=',')
    baseline_mod_d = gen_modularity_densities(dyn_mat, baseline_cluster_result, typ=0)
    print("D for Spectral: {}".format(baseline_mod_d))
    baseline_nmi = []
    for k in range(1, len(baseline_cluster_result)):
        [_, tmp] = get_mutual_information(baseline_cluster_result[k - 1], baseline_cluster_result[k])
        baseline_nmi.append(tmp)
    print("NMI for Spectral: {}".format(baseline_nmi))
    test_types = ['CrENMF2']
    markers = ['o', 's', '^', '.', ',', 'v', 's', 'p', 'h', '+']
    results_folders = [results_folder_base + '/GrENMF2/5/', results_folder_base + '/CrENMF2/3_3/']
    fig_d = plt.figure(figsize=(20, 8))
    fig_nmi = plt.figure(figsize=(20, 8))
    plt.figure(fig_d)
    plt.plot(baseline_mod_d, label='Spectral', marker=markers[-1])
    plt.title('D-values of Different Clustering Algorithms', fontsize=30)
    plt.figure(fig_nmi)
    plt.plot(list(range(1, len(baseline_nmi) + 1)), baseline_nmi, label='Spectral', marker=markers[-1])
    plt.title('NMIs of Different Clustering Algorithms', fontsize=30)
    for (m, results_folder) in enumerate(results_folders):
        lookup_keys = ['GrENMF', 'CrENMF']
        cluster_result_path = '{}/cluster_result.csv'.format(results_folder)
        cluster_result = np.loadtxt(cluster_result_path, delimiter=',')
        mod_d = gen_modularity_densities(dyn_mat, cluster_result, typ=0)
        print("D for {}:{}".format(lookup_keys[m], mod_d))
        plt.figure(fig_d)
        plt.plot(mod_d, label=lookup_keys[m], marker=markers[m])
        nmi = []
        for k in range(1, len(cluster_result)):
            [_, tmp] = get_mutual_information(cluster_result[k - 1], cluster_result[k])
            nmi.append(tmp)
        print("NMI for {}:{}".format(lookup_keys[m], nmi))
        plt.figure(fig_nmi)
        plt.plot(list(range(1, len(nmi) + 1)), nmi, label=lookup_keys[m], marker=markers[m])
    plt.figure(fig_d)
    plt.xlabel('Time (day)', fontsize=20)
    plt.ylabel('D-value', fontsize=20)
    plt.grid(True, which='both', color='grey', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.tick_params(axis='both', labelsize=16)
    plt.minorticks_on()
    plt.legend(fontsize=20)
    plt.figure(fig_nmi)
    plt.xlabel('Time (day)', fontsize=20)
    plt.ylabel('NMI', fontsize=20)
    plt.grid(True, which='both', color='grey', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.tick_params(axis='both', labelsize=16)
    plt.minorticks_on()
    plt.legend(fontsize=20)

    plt.show()
