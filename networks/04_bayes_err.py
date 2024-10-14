import numpy as np


def get_bayes_syn(p):
    """
    Calculate bayes risk on markov chain
    """

    tr = np.array([[p, 1-p], [1-p, p]])
    pi = np.array([1, 0])

    err = []
    for i in range(10):
        err.append(np.min(pi))
        pi = pi @ tr

    print(p, np.mean(err))


def get_bayes_syn_gamm(p):
    """
    Calculate bayes risk on markov chain
    """

    tr = np.array([[p, 1-p], [1-p, p]])
    pi = np.array([1, 0])

    err = []
    gamma = 0.95
    for i in range(1000):
        err.append(np.min(pi) * gamma ** i)
        pi = pi @ tr

    norm = (1 - gamma) / (1 - gamma ** 1000)
    print(p, np.sum(err) * norm)

# get_bayes_syn(0.1)
# get_bayes_syn(0.2)

get_bayes_syn_gamm(0.1)
