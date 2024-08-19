import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm.auto as tqdm

"""
We must make a sequence of predictions using (Z<t, a<t)

Similar to scenario 3, X_t = 1 (constant)
and Y_t = 0 or 1

There is MDP such that
- Transition matrix:  T(Y_{t-1}, h_t) -> Y_t
- Loss             :  L(Y_t, h_t) -> |Y_t - h_t|

The goal is a to find a sequence h_{>t} that minimizes the expected discounted loss
"""



class Data_Scenario4():
    def __init__(self, p=0.9, τ=30, max_t=1000):
        self.p = [[p, 1-p], [1-p, p]]
        self.τ = τ
        self.max_t = max_t
        gamma = 0.9
        self.gamma_vals = np.array([gamma**i for i in range(max_t)])


class MDP():
    def __init__(self):
        α = 0.1
        β = 0.1

        # T(a_t, s_t) = P(s_{t+1} | s_t, a_t)
        self.T = np.array([
            [[α, 1-α],
             [1-α, α]],
            [[β, 1-β],
             [1-β, β]],
        ])
        self.state = np.random.choice([0, 1])

    def step(self, action):
        y_prev = self.state
        action = int(action)

        y_next = np.random.choice([0, 1], p=self.T[action, y_prev])
        loss = np.abs(action - y_next)
        self.state = y_next

        return y_prev, loss


def get_data(t):
    mdp = MDP()

    Z_t = []
    A_t = []
    L_t = []
    for i in range(t):
        action = np.random.choice([0, 1])
        z_true, loss = mdp.step(action)
        Z_t.append(z_true)
        A_t.append(action)
        L_t.append(loss)
    return Z_t, A_t, L_t


def erm_mdp(Z_t, ntest):
    p_hat = np.mean(Z_t)
    y_hat = p_hat > 0.5
    return np.ones(ntest) * y_hat


def prospective_mdp(Z_t, A_t, L_t, ntest):

    def solve_MDP(T, R, γ=0.9):
        # Q(a, s)
        Q_vals = np.zeros((2, 2)) + 10
        for iter in range(1000):
            Q_vals_new = np.zeros_like(Q_vals)
            for s in range(2):
                for a in range(2):
                    Q_new = 0
                    for s_prime in range(2):
                        Q_new += T[a, s, s_prime] * (R[a, s, s_prime] + γ * np.min(Q_vals[:, s_prime]))
                    Q_vals_new[a, s] = Q_new
            
            # Check for convergence
            if np.max(np.abs(Q_vals - Q_vals_new)) < 1e-5:
                break
            Q_vals = Q_vals_new
        return Q_vals

    #T[a, s, s']
    T_hat = np.zeros((2, 2, 2)) + 1

    # L[a, s, s']
    L_hat = np.zeros((2, 2, 2))

    for i in range(len(Z_t)-1):
        a_t = A_t[i]
        z_t = Z_t[i]
        z_t1 = Z_t[i+1]

        T_hat[a_t, z_t, z_t1] += 1
        L_hat[a_t, z_t, z_t1] += L_t[i]

    L_hat = L_hat / T_hat
    T_hat = T_hat / np.sum(T_hat, axis=2, keepdims=True)

    Q_star = solve_MDP(T_hat, L_hat)

    preds = []
    z_t = Z_t[-1]
    z_dist = np.array([[1 - z_t, z_t]])

    for s in range(ntest):

        # Find current action for distribution over states z_dst
        h_s = np.argmin(z_dist @ Q_star.T)
        preds.append(h_s)

        # update distribution over state
        z_dist = z_dist @ T_hat[h_s]



    return np.array(preds)


def eval_mdp(pred, z_t):
    mdp = MDP()
    gamma = 0.9
    mdp.state = z_t

    total_loss = 0
    for i in range(len(pred)):
        _, loss = mdp.step(pred[i])
        total_loss = total_loss + (gamma**i) * loss

    total_loss = (1 - gamma) * total_loss

    return total_loss



def main():

    seeds = 100000
    run_t = 50
    max_t = run_t + 100
    times = np.arange(5, run_t-1)

    all_pr, all_erm = [], []
    for t in tqdm.tqdm(times):

        pr_loss, erm_loss = [], []
        for it in range(seeds):
            Z_t, A_t, L_t = get_data(t)
            ntest = max_t - len(Z_t)

            erm_pred = erm_mdp(Z_t, ntest)
            pr_pred = prospective_mdp(Z_t, A_t, L_t, ntest)

            erm_loss.append(eval_mdp(erm_pred, Z_t[-1]))
            pr_loss.append(eval_mdp(pr_pred, Z_t[-1]))


        all_pr.append(pr_loss)
        all_erm.append(erm_loss)

    all_erm = np.array(all_erm)
    all_pr = np.array(all_pr)

    info = {
        'all_erm': all_erm,
        'all_pr': all_pr,
    }
    np.save("data/scenario4_disc.npy", info, allow_pickle=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set(context='poster',
            style='ticks',
            font_scale=0.65,
            rc={'axes.grid':True,
                'grid.color':'.9',
                'grid.linewidth':0.75})

    plt.title("Dependent samples from MDP")
    plt.xlabel("Number of samples / Time (t)")
    plt.ylabel("Discounted Prospective risk")

    plt.plot(times, np.mean(all_erm, axis=1), label="ERM")
    plt.plot(times, np.mean(all_pr, axis=1), label="Prospective")


    plt.legend(['Maximum likelihood estimator',
                'Time-aware empirical risk minimization',
               ])

    plt.show()
    plt.savefig("plots/scenario4_disc.pdf", bbox_inches='tight')



if __name__ == "__main__":
    main()