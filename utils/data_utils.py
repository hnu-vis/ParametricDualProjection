import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

eps = torch.finfo(torch.float32).eps


def load_json(path):
    """Load JSON data from a file.
    
    Args:
        path: Path to the JSON file to be loaded.
    
    Returns:
        dict or list: Parsed JSON data.
    
    Example:
        >>> data = load_json("config.json")
    """
    with open(path, "r") as f:
        return json.load(f)


def dump_json(obj, path):
    """Save Python object to a JSON file.
    
    Args:
        obj: Python object to be serialized (must be JSON-serializable).
        path: Destination file path.
    
    Example:
        >>> dump_json({"key": "value"}, "output.json")
    """
    with open(path, "w") as f:
        json.dump(obj, f)


def to_numpy(data):
    """Convert various data types to NumPy array.
    
    Supports conversion from: pandas DataFrame/Series, NumPy array, 
    PyTorch Tensor, and Python lists.
    
    Args:
        data: Input data to be converted.
    
    Returns:
        np.ndarray: Converted NumPy array.
    
    Raises:
        TypeError: If input type is not supported.
    
    Example:
        >>> tensor = torch.tensor([1, 2, 3])
        >>> arr = to_numpy(tensor)
    """
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        return data.to_numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, torch.Tensor):
        return data.cpu().detach().numpy()
    elif isinstance(data, list):
        return np.array(data)
    else:
        raise TypeError(f"Unsupported Data Type: {type(data)}")


def to_tensor(data):
    """Convert various data types to PyTorch Tensor.
    
    Supports conversion from: pandas DataFrame/Series, NumPy array, 
    PyTorch Tensor, and Python lists.
    
    Args:
        data: Input data to be converted.
    
    Returns:
        torch.Tensor: Converted tensor.
    
    Raises:
        TypeError: If input type is not supported.
    
    Example:
        >>> arr = np.array([1, 2, 3])
        >>> tensor = to_tensor(arr)
    """
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        return torch.tensor(data.to_numpy())
    elif isinstance(data, np.ndarray):
        return torch.tensor(data)
    elif isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, list):
        return torch.tensor(data)
    else:
        raise TypeError(f"Unsupported Data Type: {type(data)}")


def setup_seed(seed):
    """Set random seed for reproducibility across multiple libraries.
    
    Args:
        seed: Integer seed value.
    
    Note:
        Affects PyTorch (CPU/CUDA) and NumPy random number generators.
    
    Example:
        >>> setup_seed(42)  # For reproducible results
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def min_max_normalize(X: torch.Tensor, a: float = 0, b: float = 1.0, 
                     eps: float = 1e-12, tol: float = 1e-6) -> torch.Tensor:
    """Normalize tensor values to specified range [a, b] using min-max scaling.
    
    Args:
        X: Input tensor to normalize.
        a: Minimum value of output range (default: 0).
        b: Maximum value of output range (default: 1.0).
        eps: Small value to prevent division by zero.
        tol: Tolerance for constant value detection.
    
    Returns:
        torch.Tensor: Normalized tensor with values in [a, b].
    
    Note:
        Returns middle of range if input has constant values (within tolerance).
    
    Example:
        >>> x = torch.tensor([1.0, 2.0, 3.0])
        >>> normalized = min_max_normalize(x)
    """
    X_min = X.min()
    X_max = X.max()

    if torch.abs(X_max - X_min) < tol:
        return torch.full_like(X, (b - a) / 2)

    normalized = a + (b - a) * (X - X_min) / (X_max - X_min + eps)
    return normalized


def z_score_standardize(X: torch.Tensor) -> torch.Tensor:
    """Standardize tensor using z-score normalization (mean=0, std=1).
    
    Args:
        X: Input tensor to standardize.
    
    Returns:
        torch.Tensor: Standardized tensor.
    
    Note:
        Handles constant features by setting std=1 to avoid division by zero.
    
    Example:
        >>> x = torch.tensor([1.0, 2.0, 3.0])
        >>> standardized = z_score_standardize(x)
    """
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True)

    std = torch.where(std == 0, torch.tensor(1.0, device=X.device), std)

    return (X - mean) / std
