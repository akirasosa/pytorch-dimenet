import copy
import dataclasses
from functools import cached_property
from logging import getLogger, FileHandler
from multiprocessing import cpu_count
from os import cpu_count
from pathlib import Path
from time import time
from typing import Callable, List, Dict, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_ranger import Ranger
from sklearn.model_selection import KFold
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
from torch_optimizer import RAdam

from dimenet.loader import get_loader, AtomsBatch
from dimenet.logging import configure_logging
from dimenet.loss import mae_loss
from dimenet.modules.dimenet import DimeNet
from dimenet.params import ModuleParams, Params
from mylib.torch.data.dataset import PandasDataset
from mylib.torch.tools.ema.utils import update_ema


@dataclasses.dataclass(frozen=True)
class Metrics:
    lr: float
    loss: float
    lmae: float


class PLModule(pl.LightningModule):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.hparams = hparams
        self.model = DimeNet(num_targets=1)

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.best: float = float('inf')
        self.ema_model: Optional[nn.Module] = None

    def setup(self, stage: str):
        df = pd.read_parquet(self.hp.db_path)

        folds = KFold(
            n_splits=self.hp.n_splits,
            random_state=self.hp.seed,
            shuffle=True,
        )
        train_idx, val_idx = list(folds.split(df))[self.hp.fold]

        self.train_dataset = PandasDataset(df.iloc[train_idx])
        self.val_dataset = PandasDataset(df.iloc[val_idx])

    def on_train_start(self) -> None:
        super(PLModule, self).on_train_start()
        # Init ema model.
        if self.hp.ema_decay is not None:
            self.ema_model = copy.deepcopy(self.model)
            for p in self.ema_model.parameters():
                p.requires_grad_(False)

    def optimizer_step(
            self,
            epoch: int,
            batch_idx: int,
            optimizer: Optimizer,
            optimizer_idx: int,
            second_order_closure: Optional[Callable] = None,
            on_tpu: bool = False,
            using_native_amp: bool = False,
            using_lbfgs: bool = False,
    ) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, second_order_closure)
        if self.ema_model is not None:
            update_ema(self.ema_model, self.model, self.hp.ema_decay)

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        result = self.step(batch, prefix='train')
        return {
            'loss': result['train_loss'],
            **result,
        }

    def validation_step(self, batch, batch_idx):
        result = self.step(batch, prefix='val')

        if self.eval_ema:
            result_ema = self.step(batch, prefix='ema', model=self.ema_model)
        else:
            result_ema = {}

        return {
            **result,
            **result_ema,
        }

    def step(self, batch, prefix: str, model=None) -> Dict:
        batch = AtomsBatch(**batch)
        y_true = batch[self.hp.target].unsqueeze(-1)

        if model is None:
            y_pred = self.forward(batch)
        else:
            y_pred = model(batch)

        assert y_pred.shape == y_true.shape, f'{y_pred.shape}, {y_true.shape}'

        mae = mae_loss(y_pred, y_true)
        lmae = torch.log(mae)
        size = len(y_true)

        return {
            f'{prefix}_loss': lmae,
            f'{prefix}_mae': mae,
            f'{prefix}_size': size,
        }

    def training_epoch_end(self, outputs):
        metrics = self.__collect_metrics(outputs, 'train')
        self.__log(metrics, 'train')

        return {}

    def validation_epoch_end(self, outputs):
        metrics = self.__collect_metrics(outputs, 'val')
        self.__log(metrics, 'val')

        if self.eval_ema:
            metrics_ema = self.__collect_metrics(outputs, 'ema')
            self.__log(metrics_ema, 'ema')
        else:
            metrics_ema = None

        if metrics.loss < self.best:
            self.best = metrics.loss

        return {
            'progress_bar': {
                'val_loss': metrics.loss,
                'best': self.best,
            },
            'val_loss': metrics.loss,
            'ema_loss': metrics_ema.loss if metrics_ema is not None else None,
        }

    def __collect_metrics(self, outputs: List[Dict], prefix: str) -> Metrics:
        loss, mae = 0, 0
        total_size = 0

        for o in outputs:
            size = o[f'{prefix}_size']
            total_size += size
            loss += o[f'{prefix}_loss'] * size
            mae += o[f'{prefix}_mae'] * size
        loss = loss / total_size
        mae = mae / total_size

        # noinspection PyTypeChecker
        return Metrics(
            lr=self.trainer.optimizers[0].param_groups[0]['lr'],
            loss=loss,
            lmae=torch.log(mae),
        )

    def __log(self, metrics: Metrics, prefix: str):
        if self.global_step > 0:
            self.logger.experiment.add_scalar('lr', metrics.lr, self.current_epoch)
            for k, v in dataclasses.asdict(metrics).items():
                if k == 'lr':
                    continue
                self.logger.experiment.add_scalars(k, {
                    prefix: v,
                }, self.current_epoch)

    def train_dataloader(self):
        return get_loader(
            self.train_dataset,
            batch_size=self.hp.batch_size,
            shuffle=True,
            num_workers=cpu_count(),
            pin_memory=True,
            cutoff=5.,
        )

    def val_dataloader(self):
        return get_loader(
            self.val_dataset,
            batch_size=self.hp.batch_size,
            shuffle=False,
            num_workers=cpu_count(),
            pin_memory=True,
            cutoff=5.,
        )

    def configure_optimizers(self):
        if self.hp.optim == 'ranger':
            optim = Ranger
        elif self.hp.optim == 'radam':
            optim = RAdam
        else:
            raise Exception(f'Not supported optim: {self.hp.optim}')
        opt = optim(
            self.model.parameters(),
            lr=self.hp.lr,
            weight_decay=self.hp.weight_decay,
        )
        return [opt]

    @property
    def eval_ema(self) -> bool:
        if self.ema_model is None:
            return False
        f = self.hp.ema_eval_freq
        return self.current_epoch % f == f - 1

    @cached_property
    def hp(self) -> ModuleParams:
        return ModuleParams(**self.hparams)


def train(params: Params):
    seed_everything(params.m.seed)

    tb_logger = TensorBoardLogger(
        params.t.save_dir,
        name=f'dimenet_{params.m.target}',
        version=str(int(time())),
    )

    log_dir = Path(tb_logger.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = getLogger('lightning')
    logger.addHandler(FileHandler(log_dir / 'train.log'))
    logger.info(params.pretty())

    trainer = pl.Trainer(
        max_epochs=params.t.epochs,
        gpus=params.t.gpus,
        tpu_cores=params.t.num_tpu_cores,
        logger=tb_logger,
        precision=16 if params.t.use_16bit else 32,
        amp_level='O1' if params.t.use_16bit else None,
        resume_from_checkpoint=params.t.resume_from_checkpoint,
        weights_save_path=str(params.t.save_dir),
        checkpoint_callback=ModelCheckpoint(
            monitor='ema_loss',
            save_last=True,
            verbose=True,
        ),
    )
    net = PLModule(params.m.dict_config())

    trainer.fit(net)


if __name__ == '__main__':
    configure_logging()
    params = Params.load()
    train(params)
