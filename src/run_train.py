import dataclasses
import shutil
from os import cpu_count
from pathlib import Path
from pprint import pformat
from time import time
from typing import List, Dict, Optional, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import KFold, train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR

from dimenet.model.dimenet import DimeNet
from dimenet.train.const import EXP_DIR, QM9_DB
from dimenet.train.loader import AtomsBatch, get_loader
from mylib.torch.data.dataset import PandasDataset
from mylib.torch.optim.ranger import Ranger

# noinspection PyUnresolvedReferences
torch.backends.cudnn.deterministic = True
# noinspection PyUnresolvedReferences
torch.backends.cudnn.benchmark = True


@dataclasses.dataclass(frozen=True)
class Conf:
    db_path: str

    gpus: int = 1

    lr: float = 1e-4
    weight_decay: float = 1e-4

    use_16bit: bool = False

    batch_size: int = 64
    epochs: int = 400

    fold: int = 0
    n_splits: int = 4

    seed: int = 0

    ckpt_path: Optional[str] = None
    anneal_epochs: Optional[int] = None

    save_dir: str = str(EXP_DIR)

    def __post_init__(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def to_hparams(self) -> Dict:
        excludes = [
            'ckpt_path',
            'anneal_epochs',
        ]
        return {
            k: v
            for k, v in dataclasses.asdict(self).items()
            if k not in excludes
        }

    def __str__(self):
        return pformat(dataclasses.asdict(self))


@dataclasses.dataclass(frozen=True)
class Metrics:
    lr: float
    loss: float
    lmae: Union[float, torch.Tensor]


class Net(pl.LightningModule):
    def __init__(self, hparams, anneal_epochs: Optional[int] = None):
        super().__init__()
        self.hparams = hparams
        self.anneal_epochs = anneal_epochs
        self.model = DimeNet(
            128,
            num_blocks=6,
            num_bilinear=8,
            num_spherical=7,
            num_radial=6,
            num_targets=1,
        )

        self.best: float = float('inf')
        self.df_train: Optional[pd.DataFrame] = None
        self.df_val: Optional[pd.DataFrame] = None

    def on_train_start(self) -> None:
        if self.anneal_epochs is not None:
            self.trainer.max_epochs = self.trainer.current_epoch + self.anneal_epochs

    def forward(self, inputs):
        out = self.model(inputs)
        return out

    def training_step(self, batch, batch_idx):
        result = self.__step(batch, batch_idx, prefix='train')

        return {
            'loss': result['train_loss'],
            **result,
        }

    def validation_step(self, batch, batch_idx):
        return self.__step(batch, batch_idx, prefix='val')

    def training_epoch_end(self, outputs):
        metrics = self.__collect_metrics(outputs, 'train')
        self.__log(metrics, 'train')

        return {}

    def validation_epoch_end(self, outputs):
        metrics = self.__collect_metrics(outputs, 'val')
        self.__log(metrics, 'val')

        if metrics.loss < self.best:
            self.best = metrics.loss

        return {
            'progress_bar': {
                'val_loss': metrics.loss,
                'best': self.best,
            },
        }

    # noinspection PyUnusedLocal
    def __step(self, batch, batch_idx, prefix: str):
        batch = AtomsBatch(**batch)
        y_hat = self.forward(batch).squeeze(1)

        loss = lmae_loss(y_hat, batch.U0)
        mae = torch.abs(y_hat - batch.U0).mean()

        return {
            f'{prefix}_loss': loss,
            f'{prefix}_mae': mae,
            f'{prefix}_size': len(y_hat),
        }

    def __collect_metrics(self, outputs: List[Dict], prefix: str) -> Metrics:
        loss = 0
        mae = 0
        total_size = 0

        for o in outputs:
            loss += o[f'{prefix}_loss'] * o[f'{prefix}_size']
            mae += o[f'{prefix}_mae'] * o[f'{prefix}_size']
            total_size += o[f'{prefix}_size']

        # noinspection PyTypeChecker
        return Metrics(
            lr=self.trainer.optimizers[0].param_groups[0]['lr'],
            loss=loss / total_size,
            lmae=torch.log(mae / total_size),
        )

    def __log(self, metrics: Metrics, prefix: str):
        if self.global_step > 0:
            self.logger.experiment.add_scalar('lr', metrics.lr, self.current_epoch)
            self.logger.experiment.add_scalars(f'loss', {
                prefix: metrics.loss,
            }, self.current_epoch)
            self.logger.experiment.add_scalars(f'lmae', {
                prefix: metrics.lmae,
            }, self.current_epoch)

    def configure_optimizers(self):
        opt = Ranger(
            self.model.parameters(),
            lr=self.hp.lr,
            weight_decay=self.hp.weight_decay,
            use_gc=False,
        )

        if self.anneal_epochs is None:
            return [opt]

        sched = {
            'scheduler': CosineAnnealingLR(
                opt,
                T_max=self.steps_per_epoch * self.anneal_epochs,
                eta_min=0,
            ),
            'interval': 'step',
        }
        return [opt], [sched]

    def prepare_data(self):
        df = pd.read_parquet(self.hp.db_path, columns=[
            'R',
            'Z',
            'U0',
        ])

        # folds = KFold(n_splits=self.hp.n_splits, random_state=self.hp.seed, shuffle=True)
        # train_idx, val_idx = list(folds.split(df))[self.hp.fold]
        train_idx, val_idx = train_test_split(range(len(df)), train_size=110000, test_size=10000, shuffle=True, random_state=self.hp.seed)

        self.df_train = df.iloc[train_idx]
        self.df_val = df.iloc[val_idx]

    def train_dataloader(self):
        return get_loader(
            PandasDataset(self.df_train),
            batch_size=self.hp.batch_size,
            shuffle=True,
            num_workers=cpu_count(),
            pin_memory=True,
            cutoff=5.,
        )

    def val_dataloader(self):
        return get_loader(
            PandasDataset(self.df_val),
            batch_size=self.hp.batch_size,
            shuffle=False,
            num_workers=cpu_count(),
            pin_memory=True,
            cutoff=5.,
        )

    @property
    def hp(self) -> Conf:
        return Conf(**self.hparams)

    @property
    def steps_per_epoch(self) -> int:
        if self.trainer.train_dataloader is not None:
            return len(self.trainer.train_dataloader)
        return len(self.train_dataloader())


def lmae_loss(y_pred, y_true):
    loss = torch.abs(y_true - y_pred)
    loss = loss.mean(dim=0)
    loss = torch.log(loss)

    return loss


def main(conf: Conf):
    model = Net(
        conf.to_hparams(),
        anneal_epochs=conf.anneal_epochs,
    )

    logger = TensorBoardLogger(
        conf.save_dir,
        name='mol',
        version=str(int(time())),
    )

    # Copy this script to log_dir
    log_dir = Path(logger.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    shutil.copy(Path(__file__), log_dir)

    trainer = pl.Trainer(
        max_epochs=conf.epochs,
        gpus=conf.gpus,
        # num_tpu_cores=1,
        logger=logger,
        precision=16 if conf.use_16bit else 32,
        resume_from_checkpoint=conf.ckpt_path,
        weights_summary='top',
    )
    trainer.fit(model)


if __name__ == '__main__':
    print('Train DimeNet to predict U0.')

    main(Conf(
        db_path=str(QM9_DB),

        lr=1e-4,
        batch_size=32,

        # ckpt_path=str(EXP_DIR / 'mol/version_1586613839/checkpoints/epoch=389.ckpt'),
        # anneal_epochs=int(389 * 0.72),
    ))
