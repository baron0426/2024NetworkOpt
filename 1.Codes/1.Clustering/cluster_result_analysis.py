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
    for test_type in test_types:
        fig_d = plt.figure(figsize=(25, 21))
        fig_nmi = plt.figure(figsize=(30, 24))

        results_folder = results_folder_base + test_type + '/'
        for (m, param) in enumerate(os.listdir(results_folder)):
            ax_d = fig_d.add_subplot(3, 3, m+1)
            ax_nmi = fig_nmi.add_subplot(3, 3, m+1)
            ax_d.plot(baseline_mod_d, label='Spectral', marker=markers[-1])
            ax_nmi.plot(list(range(1,len(baseline_nmi)+1)), baseline_nmi, label='Spectral', marker=markers[-1])
            cluster_result_path = '{}/{}/{}/cluster_result.csv'.format(results_folder_base, test_type, param)
            cluster_result = np.loadtxt(cluster_result_path, delimiter=',')
            # mod = gen_modularity_densities(dyn_mat, cluster_result)
            # plt.figure(fig_r)
            # mod = [x if x < 10 else 10 for x in mod]
            # plt.plot(mod, label=param, marker=markers[m])
            mod_d = gen_modularity_densities(dyn_mat, cluster_result, typ=0)
            print("D for {}:{}".format(param, mod_d))
            # [beta_val, gamma_val] = param.split('_')
            # "β="+str(int(beta_val)/10)+",γ="+str(int(gamma_val)/10)
            ax_d.plot(mod_d, label="α=" + str(int(param)/10), marker=markers[m])
            nmi = []
            for k in range(1, len(cluster_result)):
                [_, tmp] = get_mutual_information(cluster_result[k - 1], cluster_result[k])
                nmi.append(tmp)
            print("NMI for {}:{}".format(param, nmi))
            # plt.figure(fig_nmi)
            ax_nmi.plot(list(range(1,len(nmi)+1)), nmi, label="α=" + str(int(param)/10), marker=markers[m])
            ax_nmi.legend(fontsize=20)
            ax_nmi.grid(True, which='both', color='grey', linestyle='-', linewidth=0.5, alpha=0.3)
            # Add grid lines
            ax_nmi.tick_params(axis='both', labelsize=16)
            ax_nmi.minorticks_on()
            ax_nmi.set_xlabel('Time (day)', fontsize=20)
            ax_nmi.set_ylabel('NMI', fontsize=20)
            ax_d.legend(fontsize=20)
            ax_d.set_xlabel('Time (day)', fontsize=20)
            ax_d.set_ylabel('D-value', fontsize=20)
            ax_d.grid(True, which='both', color='grey', linestyle='-', linewidth=0.5, alpha=0.3)
            ax_d.tick_params(axis='both', labelsize=16)
            ax_d.minorticks_on()
        # Set an overall title for the figure

        fig_d.suptitle('D-values of {} and Spectral Clustering'.format(test_type[:6]), fontsize=30)
        fig_nmi.suptitle('NMIs of {} and Spectral Clustering'.format(test_type[:6]), fontsize=30)

        plt.tight_layout()
        plt.show()
        # plt.figure(fig_r)
        # plt.legend()
        # plt.title(test_type)
        # plt.xlabel('Time Step (day)')
        # plt.ylabel('R-value')

        # plt.figure(fig_d)
        # plt.legend()
        # plt.title(test_type)
        # plt.xlabel('Time Step (day)')
        # plt.ylabel('D-value')
        #
        # plt.figure(fig_nmi)
        # plt.legend()
        # plt.title(test_type)
        # plt.xlabel('Time Step (day)')
        # plt.ylabel('NMI')

        plt.show()
