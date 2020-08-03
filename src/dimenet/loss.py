import torch


def mae_loss(y_pred, y_true):
    err = torch.abs(y_true - y_pred)
    mae = err.mean()
    return mae
