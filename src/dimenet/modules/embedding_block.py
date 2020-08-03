import torch
import torch.nn as nn

from mylib.torch.nn.modules.dense import Dense


class EmbeddingBlock(nn.Module):
    def __init__(self, emb_size, num_radial, activation=None):
        super(EmbeddingBlock, self).__init__()
        self.embedding = nn.Embedding(100, emb_size, padding_idx=0)
        self.dense_rbf = Dense(num_radial, emb_size, activation=activation)
        self.dense = Dense(emb_size * 3, emb_size, activation=activation)

    def forward(self, inputs):
        Z, rbf, idnb_i, idnb_j = inputs

        rbf = self.dense_rbf(rbf)
        x = self.embedding(Z)

        x1 = x[idnb_i]
        x2 = x[idnb_j]

        x = torch.cat((x1, x2, rbf), dim=-1)
        x = self.dense(x)

        return x
