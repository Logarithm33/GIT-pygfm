import json
import os
import os.path as osp
import random
from pathlib import Path

import numpy as np
import torch


def mask2idx(mask):
    """Convert boolean mask to index tensor."""
    return torch.where(mask == True)[0]


def idx2mask(idx, num_instances):
    """Convert index tensor to boolean mask of length ``num_instances``."""
    mask = torch.zeros(num_instances, dtype=torch.bool)
    mask[idx] = 1
    return mask


def get_device_from_model(model):
    """Get the device of the first parameter of a model."""
    return next(model.parameters()).device


def check_path(path):
    """Create directory path if it doesn't exist."""
    if not osp.exists(path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def get_n_params(model):
    """Return number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def combine_dicts(dicts, decimals=2):
    """Combine list of dicts into mean/std dict."""
    result = {}
    for d in dicts:
        for key, value in d.items():
            if key not in result:
                result[key] = []
            result[key].append(value)

    final_result = {}
    for key, value in result.items():
        if isinstance(value[0], list):
            final_result[key + '_mean'] = np.round(np.mean(value, axis=0), decimals)
            final_result[key + '_std'] = np.round(np.std(value, axis=0), decimals)
        else:
            final_result[key + '_mean'] = np.round(np.mean(value), decimals)
            final_result[key + '_std'] = np.round(np.std(value), decimals)

    return final_result


def get_scheduler(optimizer, use_scheduler=True, epochs=1000):
    """Cosine annealing scheduler."""
    if use_scheduler:
        scheduler = lambda epoch: (1 + np.cos(epoch * np.pi / epochs)) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    else:
        scheduler = None
    return scheduler


def load_params(model, path):
    """Load state dict into model."""
    model.load_state_dict(torch.load(path))
    return model


def freeze_params(model):
    """Freeze all parameters of a model."""
    for param in model.parameters():
        param.requires_grad = False
    return model
