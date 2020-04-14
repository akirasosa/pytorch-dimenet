from os import cpu_count

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

from dimenet.model.dimenet import DimeNet
from dimenet.train import loader
from dimenet.train.const import QM9_DB
from mylib.tools.lr_finder import LRFinder
from mylib.torch.data.dataset import PandasDataset
from mylib.torch.optim.ranger import Ranger


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DimeNet(
            128,
            num_blocks=6,
            num_bilinear=8,
            num_spherical=7,
            num_radial=6,
            num_targets=1,
        )

    def forward(self, x):
        x = self.model(x)
        return x


def get_loader():
    df = pd.read_parquet(QM9_DB, columns=[
        'R',
        'Z',
        'U0',
    ])

    folds = KFold(n_splits=4, random_state=0, shuffle=True)
    train_idx, val_idx = list(folds.split(df))[0]
    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]

    train_loader = loader.get_loader(
        PandasDataset(df_train),
        batch_size=32,
        shuffle=True,
        num_workers=cpu_count(),
        pin_memory=True,
        cutoff=5.,
    )
    val_loader = loader.get_loader(
        PandasDataset(df_val),
        batch_size=32,
        shuffle=False,
        num_workers=cpu_count(),
        pin_memory=True,
        cutoff=5.,
    )

    return train_loader, val_loader


def lmae_loss(y_pred, y_true):
    loss = torch.abs(y_true - y_pred)
    loss = loss.mean(dim=0)
    loss = torch.log(loss)

    return loss


# %%
model = Net().cuda()
opt = Ranger(
    model.parameters(),
    lr=1e-7,
    weight_decay=1e-4,
)
train_loader, val_loader = get_loader()
lr_finder = LRFinder(model, opt, lmae_loss, device="cuda")

# %%
lr_finder.range_test(train_loader, val_loader=val_loader, end_lr=3e+0, num_iter=100)

# %%
lr_finder.plot()
