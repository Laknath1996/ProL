import numpy as np
import os
import tqdm
from utils import plot_Qvalues

from env import GridWorldMDP


def q_learning_offline(env, T=100000, ignore_time=False):
    # Collect dataset
    state, t = env.reset()

    np.random.seed(42)
    experience = []
    for i in range(T):
        action = np.random.randint(0, 5)
        next_state, next_t, reward = env.step(action)
        experience.append((state, t, action, reward))
        state, t = next_state, next_t

    tsize = 1 if ignore_time else env.time_size
    Q_table = np.zeros((env.state_size, tsize, env.action_size))

    # Estimate q-values offline Q((s, t), a)
    for i in tqdm.tqdm(range(100)):
        for step in range(len(experience)-1):

            s_p, t_p, a_p, r_p = experience[step]
            s_t, t_t, a_t, r_t = experience[step+1]

            st_t = s_t[0] * env.grid_size + s_t[1]
            st_p = s_p[0] * env.grid_size + s_p[1]

            if ignore_time:
                t_p, t_t = 0, 0

            Q_table[st_p, t_p, a_p] = r_p + 0.9 * np.max(Q_table[st_t, t_t, :])

    return Q_table


# Example usage
def run_episode(env, Q_table=None, ignore_time=False, path='qlearning'):
    state, t = env.reset()
    env.render_init()
    os.makedirs(path, exist_ok=True)
    total_reward = 0

    env.render(0, total_reward, path)
    
    for i in range(60):
        # Simple random action selection
        st = state[0] * env.grid_size + state[1]
        if Q_table is None:
            action = np.random.choice(range(env.action_size))
        else:
            if ignore_time:
                action = np.argmax(Q_table[st, 0, :])
            else:
                action = np.argmax(Q_table[st, t, :])
        
        # Step through environment
        state, t, reward = env.step(action)
        total_reward += reward
        print(t, reward)
        
        env.render(action, total_reward, path)
    
    env.close()
    return total_reward


def main():
    env = GridWorldMDP()
    Q_table = q_learning_offline(env, ignore_time=False)
    run_episode(env, Q_table, ignore_time=False, path='prospective')
    plot_Qvalues(Q_table, env, [0, 10], 'prospective')
    
    env = GridWorldMDP()
    Q_table = q_learning_offline(env, ignore_time=True)
    run_episode(env, Q_table, ignore_time=True, path='qlearning')
    plot_Qvalues(Q_table, env, [0], 'qlearning')

if __name__ == "__main__":
    main()

