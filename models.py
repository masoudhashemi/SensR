from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TabluarModel(nn.Module):
    def __init__(
        self,
        cat_idx,
        num_features,
        embedding_sizes,
        layers,
        n_classes,
        dropout=False,
    ):
        # embedding_sizes: list of [num_categories, num_embedding]
        super().__init__()
        n_cont = num_features - len(cat_idx)
        self.cat_idx = cat_idx
        self.dropout = dropout
        self.num_idx = [i for i in range(num_features) if i not in cat_idx]
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(categories, size)
                for categories, size in embedding_sizes
            ]
        )
        n_emb = sum(
            e.embedding_dim for e in self.embeddings
        )  # length of all embeddings combined
        self.n_emb, self.n_cont = n_emb, n_cont
        self.lin1 = nn.Linear(self.n_emb + self.n_cont, layers[0])
        self.lin2 = nn.Linear(layers[0], layers[1])
        self.lin3 = nn.Linear(layers[1], n_classes)
        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.bn2 = nn.BatchNorm1d(layers[0])
        self.bn3 = nn.BatchNorm1d(layers[1])
        self.emb_drop = nn.Dropout(0.6)
        self.drops = nn.Dropout(0.3)

    def forward(self, x):
        x_cat = x[:, self.cat_idx].long()
        x_cont = x[:, self.num_idx]
        x1 = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        x1 = torch.cat(x1, 1)
        if self.dropout:
            x1 = self.emb_drop(x1)
        x2 = self.bn1(x_cont)
        x = torch.cat([x1, x2], 1)
        x = F.relu(self.lin1(x))
        if self.dropout:
            x = self.drops(x)
        x = self.bn2(x)
        x = F.relu(self.lin2(x))
        if self.dropout:
            x = self.drops(x)
        x = self.bn3(x)
        x = self.lin3(x)
        return x


class MLP(nn.Module):
    def __init__(self, n_features, layers, n_classes):
        super().__init__()
        self.layers = [nn.Linear(n_features, layers[0])]
        self.layers.append(nn.ReLU())
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(layers[-1], n_classes))
        self.mlp = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.mlp(x)
        return x


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def train_epoch(self, x, y, optimizer=None, loss=None, lr=0.001, retain_graph=False):
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer
        if loss is None:
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = loss

        self.optimizer.zero_grad()
        classifier_output = self.model(x)
        classifier_loss = self.loss(classifier_output, y)
        classifier_loss.backward(retain_graph=retain_graph)
        optimizer.step()
        return classifier_loss

    def forward(self, x):
        self.model.forward(x)
