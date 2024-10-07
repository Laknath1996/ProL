import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np


#### Model
class TimeEmbedding(nn.Module):
    def __init__(self, dim, dev):
        super(TimeEmbedding, self).__init__()
        self.freqs = (2 * np.pi) / (torch.arange(2, dim + 1, 2))
        self.freqs = self.freqs.unsqueeze(0)

    def forward(self, t):
        self.sin = torch.sin(self.freqs * t)
        self.cos = torch.cos(self.freqs * t)
        return torch.cat([self.sin, self.cos], dim=-1)


class ProspectiveMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, tdim=50, dev='cpu'):
        super(ProspectiveMLP, self).__init__()
        self.time_embed = TimeEmbedding(tdim, dev)
        in_dim += tdim

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, t):
        tembed = self.time_embed(t.reshape(-1, 1))
        x = torch.cat([x, tembed], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


#### Data
def online_sample_from_task_sequence(t, N=20, d=2):
    if (t // N) % 2 == 0:
        mu = np.array([1, 1])
    else:
        mu = np.array([1, -1])    
    Y = np.random.binomial(1, 0.5)
    X = np.random.multivariate_normal((-1)**(Y+1)*mu, np.eye(d))
    return X, Y

def batch_sample_from_task_sequence(n, t, N=20, d=2, seed=1996):
    if (t // N) % 2 == 0:
        mu = np.array([1, 1])
    else:
        mu = np.array([1, -1])    
    X = np.concatenate((
        np.random.multivariate_normal((-1)*mu, np.eye(d), size=n // 2),
        np.random.multivariate_normal(mu, np.eye(d), size=n // 2)))
    Y = np.concatenate((-np.zeros(n // 2), np.ones(n // 2)))
    return X, Y

def create_tf_batch(x, y, t):
    x = torch.Tensor(x).reshape(-1, 2).float()
    y = torch.Tensor(y).reshape(-1).long()
    tt = torch.Tensor(t).reshape(-1).long()
    return x, y, tt


def train_model():
    T_max = 10000
    period = 10
    n_test = 10000

    net = ProspectiveMLP(in_dim=2, out_dim=2, hidden_dim=200, tdim=50)
    opt = torch.optim.SGD(net.parameters(), lr=0.001, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    accs = []

    for t in range(T_max):

        # Train
        x, y = online_sample_from_task_sequence(t, period)
        x, y, tt = create_tf_batch(x, [y], [t])

        logits = net(x, tt)
        loss = criterion(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        # Test
        Xt, Yt = batch_sample_from_task_sequence(n_test, t, period)

        Xt, Yt, tt = create_tf_batch(Xt, Yt, [t] * n_test)

        with torch.no_grad():
            logits = net(Xt, tt)
            pred = torch.argmax(logits, dim=-1)
            acc = torch.mean((pred == Yt).float()).item()

        print("At time {}, acc: {}".format(t, acc))
        accs.append(acc)

    # Plot accs
    plt.plot(accs)
    plt.savefig('prospective_mlp.png')
    plt.show()



train_model()
