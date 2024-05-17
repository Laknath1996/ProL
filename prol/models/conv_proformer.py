import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from tqdm.auto import tqdm
from ..utils import get_dataloader
from .base_trainer import BaseTrainer
from .resnet import Model as FeatureModel
from .resnet import model_defaults as feature_model_defaults


class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, p=0.1):
        super().__init__()
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, hidden_dim, p),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model, p),
        )
        self.mha = nn.MultiheadAttention(d_model, num_heads, p)
        self.layernorm1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)

    def forward(self, x):
        attn_output, _ = self.mha(x, x, x)
        out1 = self.layernorm1(x + attn_output)
        ff_output = self.feed_forward(out1)
        out2 = self.layernorm2(out1 + ff_output)
        return out2

class Model(nn.Module):
    def __init__(self, input_size, d_model, num_heads, ff_hidden_dim, num_attn_blocks=1, num_classes=2, 
                 contextlength=200, max_len=5000):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model

        self.attention_blocks = nn.ModuleList(
            [SelfAttention(d_model, num_heads, ff_hidden_dim) for _ in range(num_attn_blocks)]
        )

        self.input_embedding = nn.Linear(input_size+1, d_model//2)
        self.layernorm = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.classifier = nn.Linear(d_model, num_classes)

        # frequency-adjusted fourier encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = 2 * math.pi / torch.arange(2, d_model//2 + 1, 2)
        ffe = torch.zeros(1, max_len, d_model//2)
        ffe[0, :, 0::2] = torch.sin(position * div_term)
        ffe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('ffe', ffe)

        # featurizer
        featurizer_kwargs = feature_model_defaults('cifar-10')
        model = FeatureModel(**featurizer_kwargs)
        self.featurizer = nn.Sequential(
            *list(model.children())[:-1]
        )
        
    def time_encoder(self, t):
        enc = torch.cat([self.ffe[:, t[i].squeeze().long(), :] for i in range(t.size(0))])
        return enc

    def forward(self, data, labels, times):  
        data = data.permute(0, 1, 4, 2, 3)
        x = torch.stack([self.featurizer(datum).squeeze() for datum in data], axis=0)

        u = torch.cat((x, labels.unsqueeze(-1)), dim=-1)
        u = self.input_embedding(u)

        t = self.time_encoder(times)

        x = torch.cat((u, t), dim=-1)

        for attn_block in self.attention_blocks:
            x = attn_block(x)
        x = torch.select(x, 1, -1)
        x = self.classifier(x)
        return x
    
def model_defaults(dataset):
    if dataset == 'mnist':
        return { 
            "input_size": 28*28,
            "d_model": 512, 
            "num_heads": 8,
            "ff_hidden_dim": 2048,
            "num_attn_blocks": 4,
            "contextlength": 200
        }
    elif dataset == 'cifar-10':
        return { 
            "input_size": 32,
            "d_model": 256, 
            "num_heads": 8,
            "ff_hidden_dim": 1024,
            "num_attn_blocks": 2, 
            "contextlength": 200
        }
    else:
        raise NotImplementedError

class Trainer(BaseTrainer):
    def __init__(self, model, dataset, args) -> None:
        super().__init__(model, dataset, args)

    def fit(self, log):
        args = self.args
        nb_batches = len(self.trainloader)
        for epoch in range(args.epochs):
            self.model.train()
            losses = 0.0
            train_acc = 0.0
            for data, time, label, target in self.trainloader:
                data = data.float().to(self.device)
                time = time.float().to(self.device)
                label = label.float().to(self.device)
                target = target.long().to(self.device)

                out = self.model(data, label, time)
                loss = self.criterion(out, target)

                self.optimizer.zero_grad()
                loss.backward()
                losses += loss.item()
                self.optimizer.step()
                train_acc += (out.argmax(1) == target).detach().cpu().numpy().mean()
                self.scheduler.step()
            
            if args.verbose and (epoch+1) % 10 == 0:
                info = {
                    "epoch" : epoch + 1,
                    "loss" : np.round(losses/nb_batches, 4),
                    "train_acc" : np.round(train_acc/nb_batches, 4)
                }
                log.info(f'{info}')

    def evaluate(self, test_dataset, verbose=False):
        testloader = get_dataloader(
            test_dataset,
            batchsize=100,
            train=False
        )
        self.model.eval()
        with torch.no_grad():
            preds = []
            truths = []
            if verbose:
                progress = tqdm(testloader)
            else:
                progress = testloader
            for data, time, label, target in progress:
                data = data.float().to(self.device)
                time = time.float().to(self.device)
                label = label.float().to(self.device)
                target = target.long().to(self.device)

                out = self.model(data, label, time)

                preds.extend(
                    out.detach().cpu().argmax(1).numpy()
                )
                truths.extend(
                    target.detach().cpu().numpy()
                )
        return np.array(preds), np.array(truths)

if __name__ == "__main__":
    # testing
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    kwargs = model_defaults("cifar-10")
    net = Model(num_classes=2, **kwargs)
    net.to(device)

    data = torch.randn((16, 201, 32, 32, 3)).to(device)
    labels = torch.randn((16, 201)).to(device)
    times = torch.randn((16, 201)).to(device)

    y = net(data, labels, times)
    print(y.shape)