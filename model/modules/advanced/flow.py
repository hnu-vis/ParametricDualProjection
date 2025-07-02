from typing import List

import torch
import torch.nn as nn
import torch.nn.init as init

from ..base.mlp import AutoMLP


class AffineCoupling(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.input_dim = input_dim
        self.x_0_dim = output_dim
        self.x_1_dim = input_dim - output_dim

        # self.s1 = AutoMLP(self.x_1_dim, self.x_0_dim, True)
        self.t1 = AutoMLP(self.x_1_dim, self.x_0_dim)

        self.s2 = AutoMLP(self.x_0_dim, self.x_1_dim, True)
        self.t2 = AutoMLP(self.x_0_dim, self.x_1_dim)

    def encode(self, X: torch.Tensor):
        X_0 = X[..., :self.x_0_dim]
        X_1 = X[..., self.x_0_dim:]

        # Y_0 = X_0
        # Y_1 = X_1 + self.t2(Y_0)
        # Y_0 = X_0
        Y_0 = X_0 + self.t1(X_1)
        Y_1 = X_1 * torch.exp(self.s2(Y_0)) + self.t2(Y_0)

        Y = torch.cat([Y_0, Y_1], dim=-1)

        return Y

    def decode(self, Y: torch.Tensor):
        Y_0 = Y[..., :self.x_0_dim]
        Y_1 = Y[..., self.x_0_dim:]

        # X_1 = Y_1 - self.t2(Y_0)
        # X_0 = Y_0
        X_1 = (Y_1 - self.t2(Y_0)) * torch.exp(-self.s2(Y_0))
        # X_0 = Y_0
        X_0 = Y_0 - self.t1(X_1)

        X = torch.cat([X_0, X_1], dim=-1)

        return X


class Shuffle(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        permutation = torch.randperm(dim)
        self.permutation = nn.Parameter(permutation, requires_grad=False)
        self.inverse_permutation = nn.Parameter(
            torch.argsort(permutation), requires_grad=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.permutation]

    def decode(self, shuffled_x: torch.Tensor) -> torch.Tensor:
        return shuffled_x[..., self.inverse_permutation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


class Swap(nn.Module):
    def __init__(self, total_dim: int, first_part_dim: int):
        super().__init__()

        self.total_dim = total_dim
        self.first_part_dim = first_part_dim
        self.second_part_dim = total_dim - first_part_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        first_part = x[..., :self.first_part_dim]
        second_part = x[..., self.first_part_dim:]

        return torch.cat([second_part, first_part], dim=-1)

    def decode(self, swapped_x: torch.Tensor) -> torch.Tensor:
        first_part = swapped_x[..., :self.second_part_dim]
        second_part = swapped_x[..., self.second_part_dim:]

        return torch.cat([first_part, second_part], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


class Flow(nn.Module):
    def __init__(self, dims: List[int], shuffle=False):
        super().__init__()

        self.in_dim = dims[0]
        self.dims = dims

        self.shuffle = shuffle

        if shuffle:
            self.flow = nn.ModuleList([
                *[
                    module
                    for dim in dims[1:-1]
                    for module in (AffineCoupling(self.in_dim, dim), Shuffle(self.in_dim))
                ],
                AffineCoupling(self.in_dim, dims[-1])
            ])
        else:
            self.flow = nn.ModuleList([
                *[
                    module
                    for dim in dims[1:-1]
                    for module in (AffineCoupling(self.in_dim, dim), Swap(self.in_dim, dim))
                ],
                AffineCoupling(self.in_dim, dims[-1])
            ])

    def encode(self, x):
        z = x
        for block in self.flow:
            z = block.encode(z)

        return z

    def decode(self, z):
        x = z
        for block in reversed(self.flow):
            x = block.decode(x)

        return x

    def forward(self, x):
        z = self.encode(x)
        x_rec = self.decode(z)

        return z, x_rec
