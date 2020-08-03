import torch.nn as nn
from timm.models.layers import mish
from torch_scatter import scatter_add

from dimenet.functional import calculate_interatomic_distances, calculate_neighbor_angles
from dimenet.loader import get_loader, AtomsBatch
from dimenet.modules.bessel_basis_layer import BesselBasisLayer
from dimenet.modules.embedding_block import EmbeddingBlock
from dimenet.modules.interaction_block import InteractionBlock
from dimenet.modules.output_block import OutputBlock
from dimenet.modules.spherical_basis_layer import SphericalBasisLayer
from mylib.torch.data.dataset import PandasDataset
from mylib.torch.nn.mish_init import init_weights


class DimeNet(nn.Module):
    def __init__(
            self,
            emb_size=128,
            num_blocks=6,
            num_bilinear=8,
            num_spherical=7,
            num_radial=6,
            cutoff=5.0,
            envelope_exponent=5,
            num_before_skip=1,
            num_after_skip=2,
            num_dense_output=3,
            num_targets=12,
            activation=mish,
            return_hidden_outputs: bool = False,
    ):
        super(DimeNet, self).__init__()
        self.num_blocks = num_blocks
        self.return_hidden_outputs = return_hidden_outputs

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

        init_weights(self)

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
        Dij = calculate_interatomic_distances(R, idnb_i, idnb_j)
        rbf = self.rbf_layer(Dij)

        # Calculate angles
        A_ijk = calculate_neighbor_angles(R, id3dnb_i, id3dnb_j, id3dnb_k)
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
    # %%
    import pandas as pd
    from dimenet.const import QM9_DB

    # %%
    df = pd.read_parquet(QM9_DB, columns=[
        'R',
        'Z',
        'U0',
    ])
    dataset = PandasDataset(df)

    # %%
    loader = get_loader(dataset, batch_size=2, shuffle=False)
    model = DimeNet(
        128,
        num_blocks=6,
        num_bilinear=8,
        num_spherical=7,
        num_radial=6,
        num_targets=3,
        return_hidden_outputs=True,
    )

    for batch in loader:
        batch = AtomsBatch.from_dict(batch, device='cpu')
        outputs = model(batch)
        print(outputs.shape)
        print(batch.R.shape)
        break
