import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

from dimenet.model.bessel_basis_layer import BesselBasisLayer
from dimenet.model.embedding_block import EmbeddingBlock
from dimenet.model.interaction_block import InteractionBlock
from dimenet.model.output_block import OutputBlock
from dimenet.model.spherical_basis_layer import SphericalBasisLayer
from dimenet.train.const import QM9_DB
from dimenet.train.loader import AtomsBatch, get_loader
from mylib.torch.data.dataset import PandasDataset
from mylib.torch.nn.activations import mish


def _calculate_interatomic_distances(R, idx_i, idx_j):
    Ri = R[idx_i]
    Rj = R[idx_j]
    # ReLU prevents negative numbers in sqrt
    Dij = torch.sqrt(F.relu(torch.sum((Ri - Rj) ** 2, -1)))
    return Dij


def _calculate_neighbor_angles(R, id3_i, id3_j, id3_k):
    """Calculate angles for neighboring atom triplets"""
    Ri = R[id3_i]
    Rj = R[id3_j]
    Rk = R[id3_k]
    R1 = Rj - Ri
    R2 = Rk - Ri
    x = torch.sum(R1 * R2, dim=-1)
    y = torch.cross(R1, R2)
    y = torch.norm(y, dim=-1)
    angle = torch.atan2(y, x)
    return angle


class DimeNet(nn.Module):
    def __init__(
            self,
            emb_size,
            num_blocks,
            num_bilinear,
            num_spherical,
            num_radial,
            cutoff=5.0,
            envelope_exponent=5,
            num_before_skip=1,
            num_after_skip=2,
            num_dense_output=3,
            num_targets=12,
            activation=mish,
    ):
        super(DimeNet, self).__init__()
        self.num_blocks = num_blocks
        self.rbf_layer = BesselBasisLayer(
            num_radial,
            cutoff=cutoff,
            envelope_exponent=envelope_exponent,
        )
        self.sbf_layer = SphericalBasisLayer(
            num_spherical,
            num_radial,
            cutoff=cutoff,
            envelope_exponent=envelope_exponent,
        )
        self.emb_block = EmbeddingBlock(
            emb_size,
            num_radial=num_radial,
            activation=activation,
        )
        self.output_blocks = nn.ModuleList([
            OutputBlock(
                emb_size,
                num_radial=num_radial,
                n_layers=num_dense_output,
                n_out=num_targets,
                activation=activation,
            )
            for _ in range(num_blocks + 1)
        ])
        self.int_blocks = nn.ModuleList([
            InteractionBlock(
                emb_size,
                num_radial=num_radial,
                num_spherical=num_spherical,
                num_bilinear=num_bilinear,
                num_before_skip=num_before_skip,
                num_after_skip=num_after_skip,
                activation=activation,
            )
            for _ in range(num_blocks)
        ])

    def forward(self, inputs: AtomsBatch):
        Z = inputs.Z
        R = inputs.R
        idnb_i = inputs.idnb_i
        idnb_j = inputs.idnb_j
        id3dnb_i = inputs.id3dnb_i
        id3dnb_j = inputs.id3dnb_j
        id3dnb_k = inputs.id3dnb_k
        id_expand_kj = inputs.id_expand_kj
        id_reduce_ji = inputs.id_reduce_ji
        batch_seg = inputs.batch_seg

        # Calculate distances
        Dij = _calculate_interatomic_distances(R, idnb_i, idnb_j)
        rbf = self.rbf_layer(Dij)

        # Calculate angles
        A_ijk = _calculate_neighbor_angles(R, id3dnb_i, id3dnb_j, id3dnb_k)
        sbf = self.sbf_layer((Dij, A_ijk, id_expand_kj))

        # Embedding block
        x = self.emb_block((Z, rbf, idnb_i, idnb_j))
        P = self.output_blocks[0]((x, rbf, idnb_i))

        # Interaction blocks
        for i in range(self.num_blocks):
            x = self.int_blocks[i]((x, rbf, sbf, id_expand_kj, id_reduce_ji))
            P += self.output_blocks[i + 1]([x, rbf, idnb_i])

        P = scatter_add(P, batch_seg, dim=0)
        # P = torch.zeros((n_batch, P.size(1))) \
        #     .type_as(P) \
        #     .scatter_add(0, batch_seg.unsqueeze(-1).expand_as(P), P)

        return P


# %%
if __name__ == '__main__':
    import pandas as pd

    # %%
    dataset = PandasDataset(pd.read_parquet(QM9_DB))

    # %%
    loader = get_loader(dataset, batch_size=2, shuffle=False, cutoff=5.)
    model = DimeNet(
        128,
        num_blocks=6,
        num_bilinear=8,
        num_spherical=7,
        num_radial=6,
        num_targets=1,
    ).cuda()

    for batch in loader:
        batch = AtomsBatch.from_dict(batch, device='cuda')
        out = model(batch)
        print(out.shape)
        print(batch.U0.shape)
        break
