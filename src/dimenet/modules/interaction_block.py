import torch
import torch.nn as nn
from torch_scatter import scatter_add

from mylib.torch.nn.modules.dense import Dense


class ResidualLayer(nn.Module):
    def __init__(self, units, **kwargs):
        super(ResidualLayer, self).__init__()
        self.dense_1 = Dense(units, units, **kwargs)
        self.dense_2 = Dense(units, units, **kwargs)

    def forward(self, inputs):
        x = inputs + self.dense_2(self.dense_1(inputs))
        return x


class InteractionBlock(nn.Module):
    def __init__(self, emb_size, num_radial, num_spherical, num_bilinear, num_before_skip, num_after_skip,
                 activation=None):
        super(InteractionBlock, self).__init__()
        self.emb_size = emb_size
        self.num_bilinear = num_bilinear

        self.dense_rbf = Dense(num_radial, emb_size, bias=False)
        self.dense_sbf = Dense(num_radial * num_spherical, num_bilinear, bias=False)

        self.dense_ji = Dense(emb_size, emb_size, activation=activation, bias=True)
        self.dense_kj = Dense(emb_size, emb_size, activation=activation, bias=True)

        bilin_initializer = torch.empty((self.emb_size, self.num_bilinear, self.emb_size)) \
            .normal_(mean=0, std=2 / emb_size)
        self.W_bilin = nn.Parameter(bilin_initializer)

        self.layers_before_skip = nn.ModuleList([
            ResidualLayer(emb_size, activation=activation, bias=True)
            for _ in range(num_before_skip)
        ])

        self.final_before_skip = Dense(emb_size, emb_size, activation=activation, bias=True)

        self.layers_after_skip = nn.ModuleList([
            ResidualLayer(emb_size, activation=activation, bias=True)
            for _ in range(num_after_skip)
        ])

    def forward(self, inputs):
        x, rbf, sbf, id_expand_kj, id_reduce_ji = inputs
        # n_interactions = len(torch.unique(id_reduce_ji, sorted=False))

        # Initial transformation
        x_ji = self.dense_ji(x)
        x_kj = self.dense_kj(x)

        # Transform via Bessel basis
        g = self.dense_rbf(rbf)
        x_kj = x_kj * g

        # Transform via spherical basis
        sbf = self.dense_sbf(sbf)
        x_kj = x_kj[id_expand_kj]

        # Apply bilinear layer to interactions and basis function activation
        x_kj = torch.einsum("wj,wl,ijl->wi", sbf, x_kj, self.W_bilin)

        x_kj = scatter_add(x_kj, id_reduce_ji, dim=0)  # sum over messages
        # x_kj = torch.zeros((n_interactions, x_kj.size(1))) \
        #     .type_as(x_kj) \
        #     .scatter_add(0, id_reduce_ji.unsqueeze(-1).expand_as(x_kj), x_kj)

        # Transformations before skip connection
        x2 = x_ji + x_kj
        for layer in self.layers_before_skip:
            x2 = layer(x2)
        x2 = self.final_before_skip(x2)

        # Skip connection
        x = x + x2

        # Transformations after skip connection
        for layer in self.layers_after_skip:
            x = layer(x)
        return x


# %%
if __name__ == '__main__':
    import numpy as np

    # %%
    x = np.random.random((7, 64)).astype(np.float32)
    rbf = np.random.random((7, 6)).astype(np.float32)
    sbf = np.random.random((10, 42)).astype(np.float32)
    id_expand_kj = np.tile(np.arange(0, 7), 2)[:10]
    id_reduce_ji = np.tile(np.arange(0, 7), 2)[:10]

    # %%
    b = InteractionBlock(64, 6, 7, 8, 1, 2)
    b([
        torch.from_numpy(x),
        torch.from_numpy(rbf),
        torch.from_numpy(sbf),
        torch.from_numpy(id_expand_kj),
        torch.from_numpy(id_reduce_ji),
    ])
