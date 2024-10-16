import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def make_plot(info, title, figname, size=50, plot_index=None, subsample=None,
              outside_legend=False, minimal=False, discount=False):
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set(context='poster',
            style='ticks',
            font_scale=0.75,
            rc={'axes.grid':True,
                'grid.color':'.9',
                'figure.autolayout': not outside_legend,
                'grid.linewidth':0.75})
    plt.figure(figsize=(5, 5))
    plt.ylim([-0.05, 1])
    plt.title(title)

    cols = ['#377eb8', '#e41a1c', '#4daf4a', '#984ea3',
            '#ff7f00', '#ffff33', '#a65628']

    if discount:
        for m in info:
            info[m][0] = info[m][3]
            info[m][1] = info[m][4]

    methods = []
    methods_legend = []
    for m in info:
        methods.append(m)
        print(m)
        if m == 'ERM':
            methods_legend.append('Follow-the-leader')
        else:
            methods_legend.append(m)

    if plot_index is not None:
        methods = [methods[i] for i in plot_index]

    if subsample is not None:
        for m, s in subsample:
            info[methods[m]]= info[methods[m]][:,::s]

    if not minimal:
        if not discount:
            plt.ylabel("Prospective Risk")
        else:
            plt.ylabel("Discounted prospective Risk")
    plt.xlabel("Time (t)")

    for i, m in enumerate(methods):
        plt.plot(info[m][2], info[m][0], c=cols[i])

    if "_m2" in figname:
        print(figname)
        if discount and "_s" in figname:
            plt.axhline(y=0.3958, color='black', linestyle='--')
        elif "_s" in figname:
            plt.axhline(y=0.5, color='black', linestyle='--')
        else:
            plt.axhline(y=0.2768, color='black', linestyle='--')

    elif not minimal:
        plt.axhline(y=0.0, color='black', linestyle='--')


    for i, m in enumerate(methods):
        plt.scatter(info[m][2], info[m][0], c=cols[i], s=size)
        std = 2 * info[m][1] / np.sqrt(5)
        mean = info[m][0]
        plt.fill_between(info[m][2], mean-std, mean+std,
                         alpha=0.3, color=cols[i])

    if not minimal:
        if outside_legend:
            leg = plt.legend(methods_legend + ['Bayes risk'],
                       loc="upper right", markerscale=2.,
                       bbox_to_anchor=(1.82, 0.9),
                       scatterpoints=1, fontsize=15, frameon=True)
        else:
            leg = plt.legend(methods_legend + ['Bayes risk'],
                       loc="upper right", markerscale=2.,
                       scatterpoints=1, fontsize=15, frameon=True,
                       ncol=len(methods_legend)+1)

    def export_legend(legend, filename="legend.png"):
        # Earlier approach
        # fig  = legend.figure
        # fig.canvas.draw()
        # bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        # fig.savefig(filename, dpi="figure", bbox_inches=bbox)
        # legend.remove()

        from matplotlib.lines import Line2D

        learners = [text.get_text() for text in legend.get_texts()][:-1]
        print(learners)
        cols = ['#377eb8', '#e41a1c', '#4daf4a', '#984ea3',
            '#ff7f00', '#ffff33', '#a65628']

        fig_legend = plt.figure()

        legend_elements = [
            Line2D([0], [0], color=cols[i], lw=4, label=learner) for i, learner in enumerate(learners)
        ] + [
            Line2D([0], [0], color='k', lw=4, ls='--', label="Bayes Risk")
        ]
        
        fig_legend.legend(
            handles=legend_elements, 
            loc='center', 
            ncol=len(learners)+1, 
            fontsize=15, 
            frameon=True,
            markerscale=2.,
            scatterpoints=1)
        fig_legend.savefig(filename, dpi="figure", bbox_inches='tight')


    if not minimal and not outside_legend:
        export_legend(leg, filename="./figs/aug20/%s_legend.pdf" % figname)

    plt.savefig("./figs/aug20/%s.pdf" % figname, bbox_inches='tight')
    # plt.show()

def synthetic_scenario2():
    info = np.load("./metrics/syn_scenario2.pkl", allow_pickle=True)
    make_plot(info, "Synthetic Scenario 2", figname="syn_scenario2")

def synthetic_scenario3():
    info = np.load("./metrics/syn_scenario3.pkl", allow_pickle=True)
    make_plot(info, "Synthetic Scenario 3", figname="syn_scenario3")

def mnist_scenario2():
    info = np.load("./metrics/mnist_scenario2.pkl", allow_pickle=True)
    make_plot(info, "MNIST Scenario 2", figname="mnist_scenario2", minimal=True)

def mnist_scenario3():
    info = np.load("./metrics/mnist_scenario3.pkl", allow_pickle=True)
    make_plot(info, "MNIST Scenario 3", figname="mnist_scenario3", minimal=True)

def cifar_scenario2():
    info = np.load("./metrics/cifar_scenario2.pkl", allow_pickle=True)
    make_plot(info, "CIFAR Scenario 2", figname="cifar_scenario2",
              plot_index=[0, 1, 2, 3], subsample=[(2, 2), (3, 2)], minimal=True)

def cifar_scenario3():
    info = np.load("./metrics/cifar_scenario3.pkl", allow_pickle=True)
    make_plot(info, "CIFAR Scenario 3", figname="cifar_scenario3",
              plot_index=[0, 1], minimal=True)

def cifar_scenario2_all():
    info = np.load("./metrics/cifar_scenario2.pkl", allow_pickle=True)
    make_plot(info, "CIFAR Scenario 2", figname="cifar_scenario2_all",
              subsample=[(2, 2), (3, 2)], outside_legend=True)

def cifar_scenario3_all():
    info = np.load("./metrics/cifar_scenario3.pkl", allow_pickle=True)
    make_plot(info, "CIFAR Scenario 3", figname="cifar_scenario3_all", outside_legend=True)

def synthetic_scenario3_m2():
    info = np.load("./metrics/syn_scenario3_markov2.pkl", allow_pickle=True)
    make_plot(info, "Synthetic Scenario 3", figname="syn_scenario3_m2")

def mnist_scenario3_m2():
    info = np.load("./metrics/mnist_scenario3_markov2.pkl", allow_pickle=True)
    make_plot(info, "MNIST Scenario 3", figname="mnist_scenario3_m2", minimal=True)

def cifar_scenario3_m2():
    info = np.load("./metrics/cifar_scenario3_markov2.pkl", allow_pickle=True)
    make_plot(info, "CIFAR Scenario 3", figname="cifar_scenario3_m2",
              plot_index=[0, 1], minimal=True)

def synthetic_scenario3_m2_s(discount=False):
    info = np.load("./metrics/syn_scenario3_markov2_s.pkl", allow_pickle=True)
    suffix = "d" if discount else ""
    make_plot(info, "Synthetic Scenario 3", figname="syn_scenario3_m2_s" + suffix,
              discount=discount)

def mnist_scenario3_m2_s(discount=False):
    info = np.load("./metrics/mnist_scenario3_markov2_s.pkl", allow_pickle=True)
    suffix = "d" if discount else ""
    make_plot(info, "MNIST Scenario 3", figname="mnist_scenario3_m2_s" + suffix,
              minimal=True, discount=discount)

def cifar_scenario3_m2_s(discount=False):
    info = np.load("./metrics/cifar_scenario3_markov2_s.pkl", allow_pickle=True)
    suffix = "d" if discount else ""
    make_plot(info, "CIFAR Scenario 3", figname="cifar_scenario3_m2_s" + suffix,
              plot_index=[0, 1], minimal=True, discount=discount)

# synthetic_scenario2()
# synthetic_scenario3()
# synthetic_scenario3_m2()
synthetic_scenario3_m2_s()
# synthetic_scenario3_m2_s(discount=True)

# mnist_scenario2()
# mnist_scenario3()
# mnist_scenario3_m2()
# mnist_scenario3_m2_s()
# mnist_scenario3_m2_s(discount=True)

# cifar_scenario2()
# cifar_scenario3()
# cifar_scenario3_m2_s()
# cifar_scenario3_m2_s(discount=True)

# cifar_scenario2_all()
# cifar_scenario3_all()

