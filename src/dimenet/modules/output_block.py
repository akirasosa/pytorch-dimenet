import torch.nn as nn
from torch_scatter import scatter_add

from mylib.torch.nn.modules.dense import Dense
from mylib.torch.nn.modules.mlp import MLP


class OutputBlock(nn.Module):
    def __init__(self, emb_size, num_radial, n_layers, n_out=12, activation=None):
        super(OutputBlock, self).__init__()
        self.dense_rbf = Dense(num_radial, emb_size, bias=False)
        self.mlp = MLP(emb_size, n_out, n_hidden=emb_size, n_layers=n_layers, activation=activation)

    def forward(self, inputs):
        x, rbf, idnb_i = inputs
        # n_atoms = len(torch.unique(idnb_i, sorted=False))

        g = self.dense_rbf(rbf)
        x = g * x
        x = scatter_add(x, idnb_i, dim=0)
        # x = torch.zeros((n_atoms, x.size(1))) \
        #     .type_as(x) \
        #     .scatter_add(0, idnb_i.unsqueeze(-1).expand_as(x), x)
        x = self.mlp(x)
        return x
