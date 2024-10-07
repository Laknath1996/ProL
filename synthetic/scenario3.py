import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm.auto as tqdm

DISC = True
gamma = 0.9


class Data_Scenario3():
    def __init__(self, p=0.9, τ=30, max_t=1000):
        self.p = [[p, 1-p], [1-p, p]]
        self.τ = τ
        self.max_t = max_t

    def get_samples(self, t):
        y0 = np.random.choice([0, 1])
        y_seq = [y0]
        # for _ in range(self.max_t-1):
        for _ in range(t + 50 - 1):
            y_seq.append(np.random.choice([0, 1], p=self.p[y_seq[-1]]))
        y_train, y_test = y_seq[:t], y_seq[t:]
        return y_train, y_test
    
    def evaluate(self, p_hat, y_test, discount=DISC):

        p_hat = np.array(p_hat)
        y_hat = np.zeros_like(y_test)
        y_hat[p_hat > 0.5] = 1
        y_hat[ np.abs(p_hat - 0.5) < 1e-5] = 0.5

        if not DISC:
            err = np.mean(np.abs(p_hat - y_test))
        else:
            # use discounted sum
            err = 0
            for i in range(len(y_test)):
                err += gamma**i * np.abs(y_hat[i] - y_test[i])
            err = (1 - gamma) * err
        return err

    def estimator_erm(self, samples):
        p_hat = np.mean(samples)
        ntest = 20
        return np.ones(ntest) * p_hat

    def estimator_prospective(self, samples):

        # Pseudo counts (Bayesian prior)
        counts = np.array([[1, 1], [1, 1]])

        for i in range(len(samples)-1):
            counts[samples[i], samples[i+1]] += 1
        probs = counts / np.sum(counts, axis=1, keepdims=True)

        # Estimator 1
        theta = (probs[0, 0] + probs[1, 1]) / 2
        probs = np.array(
            [[theta, 1-theta],
             [1-theta, theta]])
        cur_probs = np.array(probs)

        # Estimator 2
        # cur_probs = np.array(probs)
        
        p_hat = []
        ystart = samples[-1]

        for t in range(20):
            if np.abs(cur_probs[ystart, 0] - 0.5) < 1e-5:
                ycur = np.random.choice([0, 1])
            else:
                ycur = np.argmax(cur_probs[ystart])
            cur_probs = cur_probs @ probs
            p_hat.append(ycur)

        return p_hat
    
    def get_errs(self, seeds, est=0):
        errs = []
        for t in tqdm.tqdm(range(2, self.τ-1)):
            err_seed = []

            for sd in range(seeds):
                y_train, y_test = dat.get_samples(t)

                if est == 0:
                    p_hat = dat.estimator_erm(y_train)
                else:
                    p_hat = dat.estimator_prospective(y_train)
                err_seed.append(dat.evaluate(p_hat, y_test))
            errs.append(err_seed)
        errs = np.array(errs)
        return errs




if __name__ == "__main__":
    np.random.seed(0)
    p=0.1
    seeds = 10000
    max_t = 90
    run_t = 30
    times = np.arange(2, run_t-1)

    dat = Data_Scenario3(p, run_t, max_t)
    errs_erm = dat.get_errs(seeds, 0)
    errs_prospective = dat.get_errs(seeds, 1)


    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set(context='poster',
            style='ticks',
            font_scale=0.65,
            rc={'axes.grid':True,
                'grid.color':'.9',
                'grid.linewidth':0.75})

    plt.figure(figsize=(7, 5))

    avg_erm = np.mean(errs_erm, axis=1)
    avg_pr = np.mean(errs_prospective, axis=1)
    std_erm = np.std(errs_erm, axis=1)
    std_pr = np.std(errs_prospective, axis=1)

    info = {
        'avg_erm': avg_erm,
        'avg_pr': avg_pr,
        'std_erm': std_erm,
        'std_pr': std_pr,
    }
    if DISC:
        np.save("data/scenario3_disc_4_2.npy", info, allow_pickle=True)
    else:
        np.save("data/scenario3_avg.npy", info, allow_pickle=True)

    plt.plot(times, avg_erm)
    plt.plot(times, avg_pr)

    if not DISC:
        bayes_err = 0.5
        plt.plot(times, np.ones_like(times) * bayes_err, '--', label='Bayes Optimal', color='black')
        plt.ylabel("Average Prospective risk")
        plt.ylim([0, 1])
        plt.legend(['Maximum likelihood estimator',
                    'Time-aware empirical risk minimization',
                    'Bayes risk'
                   ])
    else:
        bayes_err = 0.357
        #bayes_err = 1. / 6
        plt.plot(times, np.ones_like(times) * bayes_err, '--', label='Bayes Optimal', color='black')
        plt.ylabel("Discounted Prospective risk")
        plt.legend(['Maximum likelihood estimator',
                    'Time-aware empirical risk minimization',
                   ])

    plt.scatter(times, avg_erm)
    plt.scatter(times, avg_pr)
    plt.fill_between(times,
                     avg_erm - std_erm / np.sqrt(seeds),
                     avg_erm + std_erm / np.sqrt(seeds), alpha=0.2)
    plt.fill_between(times,
                     avg_pr - std_pr / np.sqrt(seeds),
                     avg_pr + std_pr / np.sqrt(seeds), alpha=0.2)

    plt.title("Dependent samples from Markov Chain")
    plt.xlabel("Number of samples / Time (t)")
    # plt.savefig("plots/scenario3_avg.pdf", bbox_inches='tight')

    if DISC:
        pass
        # plt.savefig("scenario3_disc.pdf", bbox_inches='tight')
    else: 
        pass
        # plt.savefig("scenario3_avg.pdf", bbox_inches='tight')
