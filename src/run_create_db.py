from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from dimenet.const import DATA_QM9_DIR, STRUCTURES_CSV, ATOM_MAP, QM9_DB


def processQM9_file(filename):
    path = DATA_QM9_DIR / filename

    stats = pd.read_csv(path, sep=' |\t', engine='python', skiprows=1, nrows=1, header=None)
    stats = stats.loc[:, 2:]
    stats.columns = ['rc_A', 'rc_B', 'rc_C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G',
                     'Cv']
    stats.astype(np.float32)

    mm = pd.read_csv(path, sep='\t', engine='python', skiprows=2, skipfooter=3, names=range(5))[4]
    if mm.dtype == 'O':
        mm = mm.str.replace('*^', 'e', regex=False).astype(float)

    return {
        **stats.iloc[0].to_dict(),
        'mulliken': mm.values.astype(np.float32),
    }


def encode_atom(atom: str) -> int:
    return ATOM_MAP[atom]


def create_db(out_path: Path):
    df = pd.read_csv(STRUCTURES_CSV)
    mol_grouped = df.groupby('molecule_name')

    def process(name):
        mol = mol_grouped.get_group(name)
        R = mol[['x', 'y', 'z']].values
        Z = mol['atom'].apply(encode_atom).values

        qm9_orig = processQM9_file(f'{name}.xyz')

        return {
            'name': name,
            'R': R.reshape(-1).astype(np.float32),
            'Z': Z.astype(np.int32),
            **qm9_orig,
        }

    results = [
        process(name)
        for name in tqdm(mol_grouped.groups.keys())
    ]
    pd.DataFrame(results).to_parquet(str(out_path))


if __name__ == '__main__':
    print(f'Create {QM9_DB}')
    create_db(QM9_DB)
