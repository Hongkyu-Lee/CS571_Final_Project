"""
APPNP implementation in DGL.
References
----------
Paper: https://arxiv.org/abs/1810.05997
Author's code: https://github.com/klicperajo/ppnp
"""
import torch.nn as nn
from dgl.nn.pytorch.conv import APPNPConv


class APPNP(nn.Module):
    def __init__(
        self,
        num_layers, 
        in_feat,
        hidden_dim,
        n_classes,
        activation,
        feat_drop=0,
        edge_drop=0,
        alpha=0.5,
        k=3,
    ):  
        super(APPNP, self).__init__()

        if type(hidden_dim)==int:
            hiddens = []
            for i in range(num_layers):
                hiddens.append(hidden_dim)

        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(in_feat, hiddens[0]))
        # hidden layers
        for i in range(1, num_layers):
            self.layers.append(nn.Linear(hiddens[i - 1], hiddens[i]))
        # output layer
        self.layers.append(nn.Linear(hiddens[-1], n_classes))
        self.activation = activation
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.propagate = APPNPConv(k, alpha, edge_drop)
        self.reset_parameters()


    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()


    def forward(self, features, g):
        # prediction step
        h = features
        h = self.feat_drop(h)
        h = self.activation(self.layers[0](h))
        for layer in self.layers[1:-1]:
            h = self.activation(layer(h))
        h = self.layers[-1](self.feat_drop(h))
        # propagation step
        h = self.propagate(g, h)
        return h