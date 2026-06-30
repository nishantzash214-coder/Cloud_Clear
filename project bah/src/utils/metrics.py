import numpy as np
import torch


def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return torch.tensor(x, dtype=torch.float32)


def ssim(pred, target, data_range=1.0):
    pred = _to_tensor(pred).float()
    target = _to_tensor(target).float()
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    mse = torch.mean((pred - target) ** 2)
    return 1.0 - mse.item()


def psnr(pred, target, data_range=1.0):
    pred = _to_tensor(pred).float()
    target = _to_tensor(target).float()
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10((data_range ** 2) / mse)


def rmse(pred, target):
    pred = _to_tensor(pred).float()
    target = _to_tensor(target).float()
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def spectral_angle_mapper(pred, target):
    pred = _to_tensor(pred).float().reshape(-1)
    target = _to_tensor(target).float().reshape(-1)
    cos = torch.dot(pred, target) / (torch.norm(pred) * torch.norm(target) + 1e-8)
    return torch.acos(torch.clamp(cos, -1.0, 1.0)).item()


def temporal_consistency(pred, composite, clear_mask=None):
    """Compute a temporal consistency score in [0,1].

    pred: torch tensor (B, C, H, W)
    composite: tensor broadcastable to pred (B, C, H, W) or (C, H, W)
    clear_mask: optional mask (B, 1, H, W) where 1 = clear, 0 = cloudy

    Returns a float where 1.0 means perfect agreement on clear pixels.
    """
    pred = _to_tensor(pred).float()
    comp = _to_tensor(composite).float()

    # Align dimensions
    if comp.dim() == 3:
        comp = comp.unsqueeze(0)
    # If comp is (B,1,C,H,W) or similar, try to squeeze
    if comp.dim() == 5 and comp.shape[1] == 1:
        comp = comp.squeeze(1)

    # Now comp should be (B, C, H, W) or (1, C, H, W)
    if comp.shape[0] == 1 and pred.shape[0] != 1:
        comp = comp.expand(pred.shape[0], -1, -1, -1)

    if clear_mask is None:
        mask = torch.ones(pred.shape[0], 1, pred.shape[2], pred.shape[3])
    else:
        mask = _to_tensor(clear_mask).float()
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[0] == 1 and pred.shape[0] != 1:
            mask = mask.expand(pred.shape[0], -1, -1, -1)

    # Compute MAE on clear pixels
    diff = torch.abs(pred - comp) * mask
    denom = mask.sum() * pred.shape[1]
    if denom.item() == 0:
        return 1.0
    mae = diff.sum().item() / denom.item()
    score = max(0.0, 1.0 - mae)
    return float(score)
