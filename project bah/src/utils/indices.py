import numpy as np
import torch


def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        t = x
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    else:
        t = torch.tensor(x)

    t = t.float()
    if t.dim() == 3:
        t = t.unsqueeze(0)
    return t


def _select_band(x: torch.Tensor, band: int) -> torch.Tensor:
    if x.dim() == 3:
        if x.shape[0] == 4:
            return x[band:band + 1]
        if x.shape[-1] == 4:
            return x[..., band:band + 1]
    if x.dim() == 4:
        if x.shape[1] == 4:
            return x[:, band:band + 1]
        if x.shape[-1] == 4:
            return x[..., band:band + 1]
    raise ValueError(f"Unsupported tensor shape for band selection: {x.shape}")


def ndvi(x):
    x = _to_tensor(x)
    nir = _select_band(x, 2)
    red = _select_band(x, 1)
    return (nir - red) / (nir + red + 1e-8)


def ndwi(x):
    x = _to_tensor(x)
    green = _select_band(x, 0)
    nir = _select_band(x, 2)
    return (green - nir) / (green + nir + 1e-8)


def savi(x):
    x = _to_tensor(x)
    nir = _select_band(x, 2)
    red = _select_band(x, 1)
    return 1.5 * (nir - red) / (nir + red + 0.5 + 1e-8)


def ndbi(x):
    x = _to_tensor(x)
    swir = _select_band(x, 3)
    nir = _select_band(x, 2)
    return (swir - nir) / (swir + nir + 1e-8)


def _to_torch(x):
    """Convert numpy array or torch tensor to torch.Tensor with shape (B, C, H, W)."""
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.tensor(x)

    t = t.float()
    if t.dim() == 3:
        t = t.unsqueeze(0)
    return t


def compute_all_indices(x):
    """Compute common spectral indices and return as torch tensors.

    Returns a dict of {name: tensor} with tensors shaped (B, 1, H, W).
    Supports numpy arrays or torch tensors with shape (B, C, H, W) or (C, H, W).
    """
    t = _to_torch(x)  # (B, C, H, W)
    green = t[:, 0:1, :, :]
    red   = t[:, 1:2, :, :]
    nir   = t[:, 2:3, :, :]
    swir  = t[:, 3:4, :, :]

    eps = 1e-8
    ndvi_t = (nir - red) / (nir + red + eps)
    ndwi_t = (green - nir) / (green + nir + eps)
    savi_t = 1.5 * (nir - red) / (nir + red + 0.5 + eps)
    ndbi_t = (swir - nir) / (swir + nir + eps)

    return {
        "ndvi": ndvi_t,
        "ndwi": ndwi_t,
        "savi": savi_t,
        "ndbi": ndbi_t,
    }


def index_error(pred, ref, threshold: float = 0.05):
    """Compute mean absolute error for each index between `pred` and `ref`.

    Returns a dict mapping index name -> {"mae": float, "pass": bool}.
    """
    p_idx = compute_all_indices(pred)
    r_idx = compute_all_indices(ref)

    errors = {}
    for name, p in p_idx.items():
        r = r_idx[name]
        mae = float(torch.mean(torch.abs(p - r)).item())
        errors[name] = {"mae": mae, "pass": mae <= threshold}
    return errors
