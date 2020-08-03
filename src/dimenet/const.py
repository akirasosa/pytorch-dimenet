from pathlib import Path

DATA_DIR = Path('../data')
EXP_DIR = Path('../experiments')

DATA_QM9_DIR = DATA_DIR / 'qm9'  # https://www.kaggle.com/zaharch/quantum-machine-9-aka-qm9
DATA_CSC_DIR = DATA_DIR / 'champs-scalar-coupling'  # https://www.kaggle.com/c/champs-scalar-coupling/data
DATA_PROCESSED_DIR = DATA_DIR / 'pytorch-dimenet'

STRUCTURES_CSV = DATA_CSC_DIR / 'structures.csv'
QM9_DB = DATA_PROCESSED_DIR / 'qm9.parquet'

ATOM_MAP = {
    'H': 1,
    'C': 6,
    'N': 7,
    'O': 8,
    'F': 9,
}
