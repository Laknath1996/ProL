import torch
import pickle
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np


class TimeEmbedding(nn.Module):
    def __init__(self, dim, dev):
        super(TimeEmbedding, self).__init__()
        self.freqs = (2 * np.pi) / (torch.arange(2, dim + 1, 2))
        self.freqs = self.freqs.unsqueeze(0)

    def forward(self, t):
        self.sin = torch.sin(self.freqs * t)
        self.cos = torch.cos(self.freqs * t)
        return torch.cat([self.sin, self.cos], dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, t):
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Linear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Linear, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)

    def forward(self, x, t):
        x = self.fc1(x)
        return x


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
def online_sample_from_task_sequence(t, N=20):
    x1 = np.random.uniform(-2, -1)
    x2 = np.random.uniform(1, 2)
    coinflip = np.random.choice([0, 1], p=[0.5, 0.5])
    x = x1 if coinflip == 0 else x2

    tind = (t % N) < (N // 2)
    if tind == 0:
        y = int(x > 0)
    else:
        y = int(x < 0)

    return x, y


def sample_test_seq(t, N=20, d=2, seed=1996):
    """
    Compute function everywhere
    """

    x1 = np.arange(-2, -1, 0.001)
    x2 = np.arange(1, 2, 0.001)
    X = np.concatenate([x1, x2])

    tind = (t % N) < (N // 2)

    if tind == 0:
        Y = X > 0
    else:
        Y = X < 0

    T = np.zeros(len(Y)) + t
    return X, Y, T


def create_tf_batch(x, y, t):
    x = torch.Tensor(x).reshape(-1, 1).float()
    y = torch.Tensor(y).reshape(-1).long()
    tt = torch.Tensor(t).reshape(-1).long()
    return x, y, tt


def train_model(model):
    period = 20
    T_max = 10000

    if model == 'prol':
        tag = 'Prospective'
        net = ProspectiveMLP(in_dim=1, out_dim=2, hidden_dim=100, tdim=50)
    elif model == 'mlp':
        tag = 'Online SGD (MLP)'
        net = MLP(in_dim=1, out_dim=2, hidden_dim=32)
    elif model == 'linear':
        tag = 'Online SGD (Linear)'
        net = Linear(in_dim=1, out_dim=2)
    else:
        raise NotImplementedError

    opt = torch.optim.SGD(net.parameters(), lr=0.05, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    errs = []

    if model == 'prol':
        iters = 50
    else:
        iters = 1

    model_preds = []
    dataset = [[], [], []]
    for t in range(T_max):

        if model != 'prol':
            x, y = online_sample_from_task_sequence(t, period)
            x, y, tt = create_tf_batch([x], [y], [t])

            logits = net(x, tt)
            loss = criterion(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
        else:
            x, y = online_sample_from_task_sequence(t, period)
            dataset[0].append(x)
            dataset[1].append(y)
            dataset[2].append(t)
            x, y, tt = create_tf_batch(dataset[0], dataset[1], dataset[2])

            for i in range(iters):
                logits = net(x, tt)
                loss = criterion(logits, y)

                opt.zero_grad()
                loss.backward()
                opt.step()

        # Test
        Xt, Yt, tt = sample_test_seq(t, period)
        Xt, Yt, tt = create_tf_batch(Xt, Yt, tt)

        with torch.no_grad():
            logits = net(Xt, tt)
            pred = torch.argmax(logits, dim=-1)
            acc = torch.mean((pred == Yt).float()).item()

            probs = torch.nn.functional.softmax(logits, dim=1)
            model_preds.append(probs.numpy())

        print("At time {}, err: {}".format(t, 1 - acc))
        errs.append(1 - acc)
    
    model_preds = np.array(model_preds)

    info = {"x": Xt.numpy(),
            "pred": model_preds,
            "errs": errs}

    # Save info
    with open('info/%s_ogd.pkl' % model, 'wb') as f:
        pickle.dump(info, f)

    # Plot accs
    plt.plot(np.arange(len(errs)), errs)
    plt.title(tag)
    plt.savefig('figs/%s_ogd.png' % model)
    plt.show()


train_model(model='prol')
# train_model(model='mlp')
# train_model(model='linear')
