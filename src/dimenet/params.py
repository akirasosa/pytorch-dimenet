import dataclasses
from typing import Optional, List

from dimenet.const import EXP_DIR, QM9_DB
from mylib.params import ParamsMixIn


@dataclasses.dataclass(frozen=True)
class TrainerParams(ParamsMixIn):
    num_tpu_cores: Optional[int] = None
    gpus: Optional[List[int]] = None
    epochs: int = 100
    use_16bit: bool = False
    resume_from_checkpoint: Optional[str] = None
    save_dir: str = str(EXP_DIR)


@dataclasses.dataclass(frozen=True)
class ModuleParams(ParamsMixIn):
    target: str = 'mu'

    lr: float = 3e-4
    weight_decay: float = 1e-4

    batch_size: int = 32

    optim: str = 'radam'

    ema_decay: Optional[float] = None
    ema_eval_freq: int = 1

    fold: int = 0
    n_splits: Optional[int] = 4

    seed: int = 0

    db_path: str = str(QM9_DB)


@dataclasses.dataclass(frozen=True)
class Params(ParamsMixIn):
    module_params: ModuleParams
    trainer_params: TrainerParams
    note: str = ''

    @property
    def m(self):
        return self.module_params

    @property
    def t(self):
        return self.trainer_params


# %%
if __name__ == '__main__':
    # %%
    p = Params.load('params/pre_train/002.yaml')
    print(p)
