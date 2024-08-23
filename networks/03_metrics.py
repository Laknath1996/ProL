import torch
import pickle
import numpy as np


def create_metrics(fnames, fout, model_names):

    infos = []
    for fname in fnames:
        with open(fname, "rb") as fp:
            info = pickle.load(fp)
        infos.append(info)

    plot_metrics = {}


    for model in range(len(infos)):

        seed_errs = []
        times = []

        for seed in range(len(infos[model])):
            info = infos[model][seed]

            errs = []
            for row in info:
                t, arr = row
                if seed == 0:
                    times.append(t)

                errs.append(np.mean(arr[t:]))
            seed_errs.append(errs)
        seed_errs = np.array(seed_errs)

        mean = np.mean(seed_errs, axis=0)
        std = np.std(seed_errs, axis=0)

        plot_metrics[model_names[model]] = np.array([mean, std, times])

    with open(fout, "wb") as fp:
        pickle.dump(plot_metrics, fp)


if __name__ == "__main__":
    model_names_s2 = ["ERM", "Prospective", "Online-SGD", "Bayesian GD"]
    model_names_s3 = ["ERM", "Prospective"]

    model_names_s2c = ["ERM", "Prospective", "Prospective (variant 2)",
                       "Online-SGD", "Bayesian GD"]
    model_names_s3c = ["ERM", "Prospective", "Online-SGD", "Bayesian GD"]
    model_names_s3c = ["ERM", "Prospective", "Prospective (variant 2)"]

    fnames_syn_s2 = ["./checkpoints/scenario2_v2/mlp_erm_errs.pkl",
                     "./checkpoints/scenario2_v2/mlp_prospective_errs.pkl",
                     "./checkpoints/scenario2_v2/mlp_ft1_errs.pkl",
                     "./checkpoints/scenario2_v2/mlp_bgd_errs.pkl"]
    fout_syn_s2 = "figures/metrics/syn_scenario2.pkl"

    fnames_syn_s3 = ["./checkpoints/scenario3_v2/erm_mlp_errs.pkl",
                     "./checkpoints/scenario3_v2/prospective_mlp_errs.pkl"] 
    fout_syn_s3 = "figures/metrics/syn_scenario3.pkl"


    fnames_mnist_s2 = ["./checkpoints/mnist_s2_v2/erm_mlp_errs.pkl",
                       "./checkpoints/mnist_s2_v2/prospective_mlp_errs.pkl",
                       "./checkpoints/mnist_s2_v2/mlp_ft1_errs.pkl",
                       "./checkpoints/mnist_s2_v2/mlp_bgd_errs.pkl"]      
    fout_mnist_s2 = "./figures/metrics/mnist_scenario2.pkl"

    fnames_mnist_s3 = ["./checkpoints/mnist_s3_v2/erm_mlp_errs.pkl",
                     "./checkpoints/mnist_s3_v2/prospective_mlp_errs.pkl"]
    fout_mnist_s3 = "figures/metrics/mnist_scenario3.pkl"

    # TODO add two more rows
    fnames_cifar_s2 = ["./checkpoints/cifar_s2/erm_cnn_errs.pkl",
                     "./checkpoints/cifar_s2/prospective_cnn_o_errs.pkl",
                     "./checkpoints/cifar_s2/prospective_cnn_i_errs.pkl",
                     "./checkpoints/cifar_s2/cnn_o_ft1_errs.pkl",
                     "./checkpoints/cifar_s2/cnn_o_bgd_errs.pkl"]
    fout_cifar_s2 = "figures/metrics/cifar_scenario2.pkl"

    fnames_cifar_s3 = ["./checkpoints/cifar_s3/erm_cnn_errs.pkl",
                     "./checkpoints/cifar_s3/prospective_cnn_o_errs.pkl",
                     "./checkpoints/cifar_s3/prospective_cnn_i_errs.pkl",
                     ]
    fout_cifar_s3 = "figures/metrics/cifar_scenario3.pkl"

    create_metrics(fnames_syn_s2, fout_syn_s2, model_names_s2)
    create_metrics(fnames_syn_s3, fout_syn_s3, model_names_s3)

    create_metrics(fnames_mnist_s2, fout_mnist_s2, model_names_s2)
    create_metrics(fnames_mnist_s3, fout_mnist_s3, model_names_s3)

    create_metrics(fnames_cifar_s2, fout_cifar_s2, model_names_s2c)
    create_metrics(fnames_cifar_s3, fout_cifar_s3, model_names_s3c)
