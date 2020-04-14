import torch
import torch.nn as nn


class Envelope(nn.Module):
    def __init__(self, exponent):
        super(Envelope, self).__init__()
        self.exponent = exponent
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, inputs):
        env_val = 1 / inputs \
                  + self.a * inputs ** (self.p - 1) \
                  + self.b * inputs ** self.p \
                  + self.c * inputs ** (self.p + 1)

        return torch.where(inputs < 1, env_val, torch.zeros_like(inputs))
