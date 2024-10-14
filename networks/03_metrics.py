import torch
import pickle
import numpy as np


def create_metrics(fnames, fout, model_names, discount=False):

    infos = []
    for fname in fnames:
        with open(fname, "rb") as fp:
            info = pickle.load(fp)
        infos.append(info)

    plot_metrics = {}


    for model in range(len(infos)):

        seed_errs = []
        disc_seed_errs = []
        times = []

        for seed in range(len(infos[model])):
            info = infos[model][seed]

            errs = []
            disc_errs = []

            for row in info:
                t, arr = row
                if seed == 0:
                    times.append(t)

                errs.append(np.mean(arr[t:]))

                # Compute discounted risk
                if discount:
                    gamma = 0.95
                    gamma_vec  = [gamma**i for i in range(len(arr[t:]))]

                    normalization = (1 - gamma**(len(arr[t:]))) / (1 - gamma)
                    disc_errs.append(np.sum(np.array(arr[t:]) * gamma_vec) / normalization)

            seed_errs.append(errs)
            if discount:
                disc_seed_errs.append(disc_errs)

        seed_errs = np.array(seed_errs)
        if discount:
            disc_seed_errs = np.array(disc_seed_errs)

        mean = np.mean(seed_errs, axis=0)
        std = np.std(seed_errs, axis=0)

        if discount: 
            disc_mean = np.mean(disc_seed_errs, axis=0)
            disc_std = np.std(disc_seed_errs, axis=0)
            plot_metrics[model_names[model]] = np.array(
                    [mean, std, times, disc_mean, disc_std])

        else:
            plot_metrics[model_names[model]] = np.array([mean, std, times])

    with open(fout, "wb") as fp:
        pickle.dump(plot_metrics, fp)


if __name__ == "__main__":
    model_names_s2 = ["ERM", "Prospective", "Online-SGD", "Bayesian GD"]
    model_names_s3 = ["ERM", "Prospective"]

    model_names_s2c = ["ERM", "Prospective", "Online-SGD", "Bayesian GD",
                       "Prospective (variant 2)", "Prospective (MLP)" ]
    model_names_s3c = ["ERM", "Prospective", "Online-SGD", "Bayesian GD"]
    model_names_s3c = ["ERM", "Prospective",
                       "Prospective (variant 2)", "Prospective (MLP)"]

    ##### Synthetic data - Scenario 2
    fnames_syn_s2 = ["./checkpoints/scenario2_v2/mlp_erm_errs.pkl",
                     "./checkpoints/scenario2_v2/mlp_prospective_errs.pkl",
                     "./checkpoints/scenario2_v2/mlp_ft1_errs.pkl",
                     "./checkpoints/scenario2_v2/mlp_bgd_errs.pkl"]
    fout_syn_s2 = "figures/metrics/syn_scenario2.pkl"

    ##### Synthetic data - Scenario 3
    fnames_syn_s3 = ["./checkpoints/scenario3_v2/erm_mlp_errs.pkl",
                     "./checkpoints/scenario3_v2/prospective_mlp_errs.pkl",
                     ] 
    fout_syn_s3 = "figures/metrics/syn_scenario3.pkl"

    ##### Synthetic data - Scenario 3 Markov 2
    fnames_syn_s3_m2 = ["./checkpoints/scenario3_markov2/erm_mlp_errs.pkl",
                        "./checkpoints/scenario3_markov2/prospective_mlp_errs.pkl"]
    fout_syn_s3_m2 = "figures/metrics/syn_scenario3_markov2.pkl"

    ##### Synthetic data - Scenario 3 Markov 2 _ reverse
    fnames_syn_s3_m2_s = ["./checkpoints/scenario3_markov2_s/erm_mlp_errs.pkl",
                        "./checkpoints/scenario3_markov2_s/prospective_mlp_errs.pkl"]
    fout_syn_s3_m2_s = "figures/metrics/syn_scenario3_markov2_s.pkl"


    ##### MNIST - Scenario 2
    fnames_mnist_s2 = ["./checkpoints/mnist_s2_v2/erm_mlp_errs.pkl",
                       "./checkpoints/mnist_s2_v2/prospective_mlp_errs.pkl",
                       "./checkpoints/mnist_s2_v2/mlp_ft1_errs.pkl",
                       "./checkpoints/mnist_s2_v2/mlp_bgd_errs.pkl"]      
    fout_mnist_s2 = "./figures/metrics/mnist_scenario2.pkl"


    # MNIST - Scenario 3
    fnames_mnist_s3 = ["./checkpoints/mnist_s3_v2/erm_mlp_errs.pkl",
                       "./checkpoints/mnist_s3_v2/prospective_mlp_errs.pkl",
                       "./checkpoints/mnist_s3_v2/mlp_ft1_errs.pkl",
                       "./checkpoints/mnist_s3_v2/mlp_bgd_errs.pkl",
                       ]
    fout_mnist_s3 = "figures/metrics/mnist_scenario3.pkl"

    # MNIST - Scenario 3 Markov 2
    fnames_mnist_s3_m2 = ["./checkpoints/mnist_s3_markov2/erm_mlp_errs.pkl", 
                        "./checkpoints/mnist_s3_markov2/prospective_mlp_errs.pkl"]
    fout_mnist_s3_m2 = "figures/metrics/mnist_scenario3_markov2.pkl"

    # MNIST - Scenario 3 Markov 2 - reverse
    fnames_mnist_s3_m2_s = ["./checkpoints/mnist_s3_markov2_s/erm_mlp_errs.pkl", 
                        "./checkpoints/mnist_s3_markov2_s/prospective_mlp_errs.pkl"]
    fout_mnist_s3_m2_s = "figures/metrics/mnist_scenario3_markov2_s.pkl"

    ##### CIFAR - Scenario 2
    fnames_cifar_s2 = ["./checkpoints/cifar_s2/erm_cnn_errs.pkl",
                       "./checkpoints/cifar_s2/prospective_cnn_o_errs.pkl",
                       "./checkpoints/cifar_s2/cnn_o_ft1_errs.pkl",
                       "./checkpoints/cifar_s2/cnn_o_bgd_errs.pkl",
                       "./checkpoints/cifar_s2/prospective_cnn_i_errs.pkl",
                       "./checkpoints/cifar_s2/prospective_mlp_errs.pkl"
                     ]
    fout_cifar_s2 = "figures/metrics/cifar_scenario2.pkl"

    ## CIFAR - Scenario 3
    fnames_cifar_s3 = ["./checkpoints/cifar_s3/erm_cnn_errs.pkl",
                       "./checkpoints/cifar_s3/prospective_cnn_o_errs.pkl",
                       "./checkpoints/cifar_s3/prospective_cnn_i_errs.pkl",
                       "./checkpoints/cifar_s3/prospective_mlp_errs.pkl",
                       "./checkpoints/cifar_s3/cnn_o_bgd_errs.pkl",
                       "./checkpoints/cifar_s3/cnn_o_ft1_errs.pkl",
                     ]
    fout_cifar_s3 = "figures/metrics/cifar_scenario3.pkl"

    # CIFAR - Scenario 3 Markov 2
    fnames_cifar_s3_m2 = ["./checkpoints/cifar_s3_markov2/erm_cnn_errs.pkl",
                         "./checkpoints/cifar_s3_markov2/prospective_cnn_o_errs.pkl"
                        ]
    fout_cifar_s3_m2 = "figures/metrics/cifar_scenario3_markov2.pkl"

    # CIFAR - Scenario 3 Markov 2 - reverse
    fnames_cifar_s3_m2_s = ["./checkpoints/cifar_s3_markov2_s/erm_cnn_errs.pkl",
                         "./checkpoints/cifar_s3_markov2_s/prospective_cnn_o_errs.pkl"
                        ]
    fout_cifar_s3_m2_s = "figures/metrics/cifar_scenario3_markov2_s.pkl"

    # create_metrics(fnames_syn_s2, fout_syn_s2, model_names_s2)
    # create_metrics(fnames_syn_s3, fout_syn_s3, model_names_s3)
    # create_metrics(fnames_syn_s3_m2, fout_syn_s3_m2, model_names_s3)
    create_metrics(fnames_syn_s3_m2_s, fout_syn_s3_m2_s, model_names_s3, discount=True)
    
    # create_metrics(fnames_mnist_s2, fout_mnist_s2, model_names_s2)
    # create_metrics(fnames_mnist_s3, fout_mnist_s3, model_names_s3)
    # create_metrics(fnames_mnist_s3_m2, fout_mnist_s3_m2, model_names_s3)
    create_metrics(fnames_mnist_s3_m2_s, fout_mnist_s3_m2_s, model_names_s3, discount=True)
    
    # create_metrics(fnames_cifar_s2, fout_cifar_s2, model_names_s2c)
    # create_metrics(fnames_cifar_s3, fout_cifar_s3, model_names_s3c)
    # create_metrics(fnames_cifar_s3_m2, fout_cifar_s3_m2, model_names_s3)
    create_metrics(fnames_cifar_s3_m2_s, fout_cifar_s3_m2_s, model_names_s3, discount=True)
