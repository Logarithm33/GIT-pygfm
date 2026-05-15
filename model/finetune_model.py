"""GIT Downstream Models — inherit ``GFMDownPromptNodeModelBase`` /
``GFMDownPromptGraphModelBase`` from pygfm."""

from typing import ClassVar

import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_mean

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
    # Embeddings (z) shape: [n, d] or [n, h, d] or [r, n, h, d]
    # Classes shape: [n] or [r, n]
    # return_head_first: if True, the first dimension of the output will be the heads, otherwise it will be the classes

    z = l2norm(z)

    ndim = z.ndim
    assert ndim in [2, 3, 4]

    if ndim == 4:
        num_runs = z.shape[0]
    else:
        num_runs = 1

    # Rearrange the embeddings as [run, head, num_nodes, dim]
    # classes as [run, num_nodes]
    if ndim == 2:
        z = rearrange(z, "n d -> 1 1 n d")
        y = rearrange(y, "n -> 1 n")
    elif ndim == 3:
        z = rearrange(z, "n h d -> 1 h n d")
        y = rearrange(y, "n -> 1 n")
    elif ndim == 4:
        z = rearrange(z, "r n h d -> r h n d")

    # Compute the class prototypes for each run.
    class_prototypes = []
    for i in range(num_runs):
        class_prototypes.append(scatter_mean(z[i], y[i], dim=1, dim_size=num_classes))
    class_prototypes = torch.stack(class_prototypes, dim=0)  # [r, h, c, d]

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
        elif ndim == 4:
            return rearrange(class_prototypes, "r h c d -> r c h d")


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

        assert ndim_query in [2, 3]
        assert ndim_proto in [2, 3, 4]

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

        total_logits = 0
        for h in range(num_heads):
            query_emb_iter = query_emb[0] if query_heads == 1 else query_emb[h]
            proto_emb_iter = proto_emb[0] if proto_heads == 1 else proto_emb[h]

            logits = distance_metric(query_emb_iter, proto_emb_iter)
            if task == 'multi':
                logits = rearrange(logits, "n (t c) -> n t c", t=n_task, c=2)
                logits = logits[:, :, 0] - logits[:, :, 1]
            total_logits += logits

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
        # query_emb in [n, d] or [n, h, d]
        # proto_emb in [c, d] or [c, h, d]

        ndim_query = query_emb.ndim
        ndim_proto = proto_emb.ndim

        assert ndim_query in [2, 3]
        assert ndim_proto in [2, 3, 4]

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

        total_logits = 0
        for h in range(num_heads):
            query_emb_iter = query_emb[0] if query_heads == 1 else query_emb[h]
            proto_emb_iter = proto_emb[0] if proto_heads == 1 else proto_emb[h]

            logits = distance_metric(query_emb_iter, proto_emb_iter)
            if task == 'multi':
                logits = rearrange(logits, "n (t c) -> n t c", t=n_task, c=2)
                logits = logits[:, :, 0] - logits[:, :, 1]
            total_logits += logits

        total_logits = total_logits / num_heads

        return total_logits
