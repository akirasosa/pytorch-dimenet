from pathlib import Path

EXP_DIR = Path('../experiments')

INPUT_DIR = Path('../input')
INPUT_QM9_DIR = INPUT_DIR / 'qm9'  # https://www.kaggle.com/zaharch/quantum-machine-9-aka-qm9
INPUT_CSC_DIR = INPUT_DIR / 'champs-scalar-coupling'  # https://www.kaggle.com/c/champs-scalar-coupling/data
INPUT_PROCESSED_DIR = INPUT_DIR / 'pytorch-dimenet'

STRUCTURES_CSV = INPUT_CSC_DIR / 'structures.csv'
QM9_DB = INPUT_PROCESSED_DIR / 'qm9.parquet'

ATOM_MAP = {
    'H': 1,
    'C': 6,
    'N': 7,
    'O': 8,
    'F': 9,
}
