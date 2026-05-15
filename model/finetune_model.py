"""
GIT Downstream Models — inherit ``GFMDownPromptNodeModelBase`` /
``GFMDownPromptGraphModelBase`` from pygfm.

Provides prototype-based few-shot / zero-shot classification alongside
standard linear classification.

Based on GIT-main ``model/finetune_model.py``, with ``einops.rearrange``
preserved.  ``torch_scatter.scatter_mean`` is replaced by a pure-PyTorch
loop (torch_scatter requires a C++ compiler unavailable in this env).
"""

from typing import ClassVar

import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from pygfm.public.model_bases import (
    GFMDownPromptNodeModelBase,
    GFMDownPromptGraphModelBase,
)


# ── Helpers (preserved from GIT original) ────────────────────────────


def l2norm(m):
    return F.normalize(m, p=2, dim=-1)


def distance_metric(m1, m2, cosine_sim=True):
    # a shape: [n, d]
    # b shape: [m, d]

    if cosine_sim:
        m1 = l2norm(m1)
        m2 = l2norm(m2)

        cross_term = torch.mm(m1, m2.t())
        logits = 2 - 2 * cross_term
    else:
        m1_sq = torch.sum(m1 ** 2, dim=1).unsqueeze(1)  # Shape: [n, 1]
        m2_sq = torch.sum(m2 ** 2, dim=1).unsqueeze(0)  # Shape: [1, m]
        cross_term = torch.mm(m1, m2.t())  # Shape: [n, m]

        logits = m1_sq + m2_sq - 2 * cross_term

    return -logits


def get_prototypes(z, y, num_classes, head_first=True):
    """Compute per-class mean embeddings.

    Equivalent to GIT original, but uses a pure-PyTorch loop instead of
    ``torch_scatter.scatter_mean`` (unavailable without C++ compiler).

    Parameters
    ----------
    z: Tensor  — ``[n, d]`` or ``[n, h, d]``
    y: Tensor  — ``[n]``
    num_classes: int
    head_first: bool
    """
    z = l2norm(z)

    ndim = z.ndim
    assert ndim in (2, 3), f"Expected z.ndim in (2,3), got {ndim}"

    if ndim == 2:
        z = rearrange(z, "n d -> 1 1 n d")
        y = rearrange(y, "n -> 1 n")
    elif ndim == 3:
        z = rearrange(z, "n h d -> 1 h n d")
        y = rearrange(y, "n -> 1 n")

    # z: [r, h, n, d], y: [r, n]
    r, h, n, d = z.shape

    # Pure-PyTorch scatter_mean replacement
    class_prototypes = torch.zeros(r, h, num_classes, d, device=z.device, dtype=z.dtype)
    for ri in range(r):
        for ci in range(num_classes):
            mask = (y[ri] == ci)
            if mask.any():
                class_prototypes[ri, :, ci, :] = z[ri, :, mask, :].mean(dim=1)

    # Rearrange output
    if ndim == 2:
        class_prototypes = rearrange(class_prototypes, "1 1 c d -> c d")
    elif ndim == 3:
        class_prototypes = rearrange(class_prototypes, "1 h c d -> h c d")

    if head_first:
        return class_prototypes
    else:
        if ndim == 2:
            return rearrange(class_prototypes, 'c d -> c 1 d')
        if ndim == 3:
            return rearrange(class_prototypes, "h c d -> c h d")


# ── Downstream Models ────────────────────────────────────────────────


class GITDownPromptNodeModel(GFMDownPromptNodeModelBase):
    """
    Downstream model for node / edge / link-prediction tasks.

    Mirrors GIT original ``TaskModel`` with pygfm base class.
    """

    gfm_family: ClassVar[str] = "git"

    def __init__(self, encoder, num_classes, device=None):
        super().__init__(device=device)
        self.encoder = encoder
        self.hidden_dim = encoder.hidden_dim
        self.num_classes = num_classes
        self.decoder = nn.Linear(self.hidden_dim, num_classes)
        self.to(self.device)

    def forward(self, x, edge_index, edge_attr=None):
        return self.encode(x, edge_index, edge_attr)

    def encode(self, x, edge_index, edge_attr=None):
        z = self.encoder.encode(x.to(self.device),
                                edge_index.to(self.device))
        return z

    def encode_graph(self, x, edge_index, batch=None, pool="mean"):
        return self.encoder.encode_graph(x, edge_index, batch, pool)

    def pooling_lin(self, x):
        return self.encoder.pooling_lin(x)

    def classify(self, x):
        return self.decoder(x)

    def get_class_prototypes(self, z, y, num_classes, head_first=False):
        return get_prototypes(z, y, num_classes, head_first=head_first)

    def proto_classify(self, query_emb, proto_emb, task='single'):
        # query_emb in [n, d] or [n, h, d]
        # proto_emb in [c, d] or [c, h, d]

        ndim_query = query_emb.ndim
        ndim_proto = proto_emb.ndim

        assert ndim_query in (2, 3)
        assert ndim_proto in (2, 3)

        if ndim_query == 2:
            query_emb = rearrange(query_emb, "n d -> n 1 d")
        if ndim_proto == 2:
            proto_emb = rearrange(proto_emb, "c d -> c 1 d")

        query_emb = rearrange(query_emb, "n h d -> h n d")
        proto_emb = rearrange(proto_emb, "c h d -> h c d")

        query_heads = query_emb.shape[0]
        proto_heads = proto_emb.shape[0]
        num_heads = max(query_heads, proto_heads)

        total_logits = 0.0
        for h in range(num_heads):
            query_emb_iter = query_emb[0] if query_heads == 1 else query_emb[h]
            proto_emb_iter = proto_emb[0] if proto_heads == 1 else proto_emb[h]

            logits = distance_metric(query_emb_iter, proto_emb_iter)
            total_logits = total_logits + logits

        total_logits = total_logits / num_heads

        return total_logits


class GITDownPromptGraphModel(GFMDownPromptGraphModelBase):
    """
    Downstream model for graph-level tasks (multi-label molecule classification).
    """

    gfm_family: ClassVar[str] = "git"

    def __init__(self, encoder, num_classes, device=None):
        super().__init__(device=device)
        self.encoder = encoder
        self.hidden_dim = encoder.hidden_dim
        self.num_classes = num_classes
        self.decoder = nn.Linear(self.hidden_dim, num_classes)
        self.to(self.device)

    def forward(self, x, edge_index, edge_attr=None):
        return self.encode(x, edge_index, edge_attr)

    def encode(self, x, edge_index, edge_attr=None):
        z = self.encoder.encode(x.to(self.device),
                                edge_index.to(self.device))
        return z

    def encode_graph(self, x, edge_index, batch=None, pool="mean"):
        return self.encoder.encode_graph(x, edge_index, batch, pool)

    def pooling_lin(self, x):
        return self.encoder.pooling_lin(x)

    def classify(self, x):
        return self.decoder(x)

    def get_class_prototypes(self, z, y, num_classes, head_first=False):
        return get_prototypes(z, y, num_classes, head_first=head_first)

    def proto_classify(self, query_emb, proto_emb, task='single'):
        ndim_query = query_emb.ndim
        ndim_proto = proto_emb.ndim

        if ndim_query == 2:
            query_emb = rearrange(query_emb, "n d -> n 1 d")
        if ndim_proto == 2:
            proto_emb = rearrange(proto_emb, "c d -> c 1 d")
        if ndim_proto == 4:
            n_task = proto_emb.shape[0]
            proto_emb = rearrange(proto_emb, "t c h d -> (t c) h d")

        query_emb = rearrange(query_emb, "n h d -> h n d")
        proto_emb = rearrange(proto_emb, "c h d -> h c d")

        query_heads = query_emb.shape[0]
        proto_heads = proto_emb.shape[0]
        num_heads = max(query_heads, proto_heads)

        total_logits = 0.0
        for h in range(num_heads):
            query_emb_iter = query_emb[0] if query_heads == 1 else query_emb[h]
            proto_emb_iter = proto_emb[0] if proto_heads == 1 else proto_emb[h]

            logits = distance_metric(query_emb_iter, proto_emb_iter)
            if task == 'multi' and ndim_proto == 4:
                logits = rearrange(logits, "n (t c) -> n t c", t=n_task, c=2)
                logits = logits[:, :, 0] - logits[:, :, 1]
            total_logits = total_logits + logits

        total_logits = total_logits / num_heads

        return total_logits
