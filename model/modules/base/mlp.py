from typing import List

import torch
import torch.nn as nn
import torch.nn.init as init


class MLP(nn.Module):
    def __init__(self, dims: List[int], last_norm: bool = False):
        super().__init__()
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if last_norm:
            layers.append(nn.LayerNorm(dims[-1]))

        self.net = nn.Sequential(*layers)

        self.apply(self.init_weight)

    def init_weight(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)


class AutoMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, last_norm: bool = False):
        super().__init__()
        self.in_dim = input_dim
        self.out_dim = output_dim

        mid_dim_1, mid_dim_2 = self.get_good_dims(input_dim, output_dim)

        self.net = nn.Sequential(
            nn.Linear(input_dim, mid_dim_1),
            nn.BatchNorm1d(mid_dim_1),
            nn.ReLU(),

            nn.Linear(mid_dim_1, mid_dim_2),
            nn.BatchNorm1d(mid_dim_2),
            nn.ReLU(),

            nn.Linear(mid_dim_2, output_dim),
            nn.LayerNorm(output_dim) if last_norm else nn.Identity()
        )

        self.apply(self.init_weight)

    def get_good_dims(self, in_dim: int, out_dim: int):
        good_dims = [2 ** i for i in (
            range(1, 15) if in_dim < out_dim else range(15, 0, -1)
        )]

        small_dim = min(in_dim, out_dim)
        large_dim = 2 * max(in_dim, out_dim)

        valid_dims = [d for d in good_dims if small_dim <= d <= large_dim]
        valid_dims_num = len(valid_dims)

        if valid_dims_num == 1:
            mid_dim_1 = mid_dim_2 = valid_dims[0]
        else:
            if in_dim >= out_dim:
                mid_dim_1 = valid_dims[0]
                mid_dim_2 = valid_dims[1 + (valid_dims_num - 1) // 2]
            else:
                mid_dim_2 = valid_dims[-1]
                mid_dim_1 = valid_dims[(valid_dims_num - 1) // 2]

        return mid_dim_1, mid_dim_2

    def init_weight(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)
