import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

eps = torch.finfo(torch.float32).eps


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def dump_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f)


def to_numpy(data):
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        return data.to_numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, torch.Tensor):
        return data.cpu().detach().numpy()
    elif isinstance(data, list):
        return np.array(data)
    else:
        raise TypeError(f"不支持的数据类型: {type(data)}")


def to_tensor(data):
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        return torch.tensor(data.to_numpy())
    elif isinstance(data, np.ndarray):
        return torch.tensor(data)
    elif isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, list):
        return torch.tensor(data)
    else:
        raise TypeError(f"不支持的数据类型: {type(data)}")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)


def min_max_normalize(X: torch.Tensor, a: float = 0, b: float = 1.0, eps: float = 1e-12, tol: float = 1e-6) -> torch.Tensor:
    X_min = X.min()
    X_max = X.max()

    # 判断是否“很接近”，使用 tol 作为阈值
    if torch.abs(X_max - X_min) < tol:
        return torch.full_like(X, (b - a) / 2)

    # 正常归一化计算
    normalized = a + (b - a) * (X - X_min) / (X_max - X_min + eps)
    return normalized


def z_score_standardize(X: torch.Tensor) -> torch.Tensor:
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True)

    std = torch.where(std == 0, torch.tensor(1.0, device=X.device), std)

    return (X - mean) / std


def apply_mean_conv1d(input_tensor: torch.Tensor, kernel_size=3):
    a = input_tensor
    if len(a.shape) == 1:
        a = a.view(1, 1, -1)

    conv = nn.Conv1d(
        in_channels=1,
        out_channels=1,
        kernel_size=kernel_size,
        padding=kernel_size//2,
        bias=False
    ).to(device=input_tensor.device, dtype=input_tensor.dtype)

    with torch.no_grad():
        conv.weight.fill_(1.0 / kernel_size)

    output = conv(a)

    return output.squeeze()
