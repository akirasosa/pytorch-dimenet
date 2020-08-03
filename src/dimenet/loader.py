import dataclasses
from typing import Dict, Union, Callable, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader


@dataclasses.dataclass
class AtomsBatch:
    batch_seg: torch.Tensor

    R: torch.Tensor
    Z: torch.Tensor

    idnb_i: torch.Tensor
    idnb_j: torch.Tensor
    id3dnb_i: torch.Tensor
    id3dnb_j: torch.Tensor
    id3dnb_k: torch.Tensor
    id_expand_kj: torch.Tensor
    id_reduce_ji: torch.Tensor

    rc_A: Optional[torch.Tensor] = None
    rc_B: Optional[torch.Tensor] = None
    rc_C: Optional[torch.Tensor] = None
    mu: Optional[torch.Tensor] = None
    alpha: Optional[torch.Tensor] = None
    homo: Optional[torch.Tensor] = None
    lumo: Optional[torch.Tensor] = None
    gap: Optional[torch.Tensor] = None
    r2: Optional[torch.Tensor] = None
    zpve: Optional[torch.Tensor] = None
    U0: Optional[torch.Tensor] = None
    U: Optional[torch.Tensor] = None
    H: Optional[torch.Tensor] = None
    G: Optional[torch.Tensor] = None
    Cv: Optional[torch.Tensor] = None
    mulliken: Optional[torch.Tensor] = None

    def __getitem__(self, item: str):
        return getattr(self, item)

    @staticmethod
    def from_dict(params: Dict, device: Union[str, torch.device]):
        return AtomsBatch(**{
            k: v.to(device)
            for k, v in params.items()
        })


def to_tensor(v: np.ndarray):
    if v.dtype in [np.float64, np.float32]:
        return torch.from_numpy(v).float()
    return torch.from_numpy(v).long()


def _concat(to_stack):
    """ function to stack (or concatentate) depending on dimensions """
    if np.asarray(to_stack[0]).ndim >= 2:
        return np.concatenate(to_stack)
    else:
        return np.hstack(to_stack)


def _bmat_fast(mats):
    new_data = np.concatenate([mat.data for mat in mats])

    ind_offset = np.zeros(1 + len(mats))
    ind_offset[1:] = np.cumsum([mat.shape[0] for mat in mats])
    new_indices = np.concatenate(
        [mats[i].indices + ind_offset[i] for i in range(len(mats))])

    indptr_offset = np.zeros(1 + len(mats))
    indptr_offset[1:] = np.cumsum([mat.nnz for mat in mats])
    new_indptr = np.concatenate(
        [mats[i].indptr[i >= 1:] + indptr_offset[i] for i in range(len(mats))])
    return sp.csr_matrix((new_data, new_indices, new_indptr))


def _restore_shape(data):
    for d in data:
        d['R'] = d['R'].reshape(-1, 3)
    return data


def get_loader(
        dataset: Dataset,
        cutoff: float = 5.,
        post_fn: Callable = to_tensor,
        **kwargs,
) -> DataLoader:
    collate_fn = AtomsCollate(cutoff=cutoff, post_fn=post_fn)
    return DataLoader(dataset, collate_fn=collate_fn, **kwargs)


@dataclasses.dataclass
class AtomsCollate:
    post_fn: Callable
    cutoff: float = 5.

    def __call__(self, examples):
        examples = _restore_shape(examples)

        data = {
            k: _concat([examples[n][k] for n in range(len(examples))])
            for k in examples[0].keys()
        }

        adj_matrices = []
        for i, e in enumerate(examples):
            R = e['R']
            D = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
            adj_matrices.append(sp.csr_matrix(D <= self.cutoff))
            adj_matrices[-1] -= sp.eye(len(e['Z']), dtype=np.bool)

        adj_matrix = _bmat_fast(adj_matrices)
        atomids_to_edgeid = sp.csr_matrix(
            (np.arange(adj_matrix.nnz), adj_matrix.indices, adj_matrix.indptr),
            shape=adj_matrix.shape)
        edgeid_to_target, edgeid_to_source = adj_matrix.nonzero()

        # Target (i) and source (j) nodes of edges
        data['idnb_i'] = edgeid_to_target
        data['idnb_j'] = edgeid_to_source

        # Indices of triplets k->j->i
        ntriplets = adj_matrix[edgeid_to_source].sum(1).A1
        id3ynb_i = np.repeat(edgeid_to_target, ntriplets)
        id3ynb_j = np.repeat(edgeid_to_source, ntriplets)
        id3ynb_k = adj_matrix[edgeid_to_source].nonzero()[1]

        # Indices of triplets that are not i->j->i
        id3_y_to_d, = (id3ynb_i != id3ynb_k).nonzero()
        data['id3dnb_i'] = id3ynb_i[id3_y_to_d]
        data['id3dnb_j'] = id3ynb_j[id3_y_to_d]
        data['id3dnb_k'] = id3ynb_k[id3_y_to_d]

        # Edge indices for interactions
        # j->i => k->j
        data['id_expand_kj'] = atomids_to_edgeid[edgeid_to_source, :].data[id3_y_to_d]
        # j->i => k->j => j->i
        data['id_reduce_ji'] = atomids_to_edgeid[edgeid_to_source, :].tocoo().row[id3_y_to_d]

        N = [len(e['Z']) for e in examples]
        data['batch_seg'] = np.repeat(np.arange(len(examples)), N)

        return {
            k: self.post_fn(v)
            for k, v in data.items()
            if k != 'name'
        }


# %%
if __name__ == '__main__':
    # %%
    from mylib.torch.data.dataset import PandasDataset
    from dimenet.const import QM9_DB

    # %%
    df = pd.read_parquet(QM9_DB, columns=[
        'R',
        'Z',
        'U0',
    ])

    # %%
    dataset = PandasDataset(df)
    loader = get_loader(dataset, batch_size=2, shuffle=False, cutoff=5.)

    for batch in loader:
        batch = AtomsBatch.from_dict(batch, device='cpu')
        print(batch.R, batch['U0'])
        break
