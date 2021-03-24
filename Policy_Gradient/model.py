import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()

    def init_weights(self):
        for layer in self.layers:
            self._init_weights(layer)

    def _init_weights(self, layer):
        nn.init.xavier_normal_(layer.weight)
        nn.init.constant_(layer.bias, 0.01)


class DenseNet(Net):
    def __init__(self, n_states, n_actions, emb_dim):
        super().__init__()
        self.layer_size = [n_states] + emb_dim + [n_actions]
        self.layers = []

        for i in range(len(self.layer_size) - 1):
            fc = nn.Linear(self.layer_size[i], self.layer_size[i + 1])
            setattr(self, f'fc{i}', fc)
            self.layers.append(fc)

        self.init_weights()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = torch.tanh(x)
        x = self.layers[-1](x)
        return x
