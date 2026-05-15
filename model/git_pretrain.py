"""
GIT Pretraining Model — inherits ``GFMPrePromptModelBase`` from pygfm.

Implements the three-task self-supervised pretraining objective:
  1. Semantic alignment loss (cross-view mutual information)
  2. Feature reconstruction loss (MSE)
  3. Topology reconstruction loss (inner-product decoder + BCE + negative sampling)
  4. Alignment regularizer (KL divergence)
"""

from copy import deepcopy
from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling

from pygfm.public.model_bases import GFMPrePromptModelBase
from model.encoder import GITEncoder
from model.decoders import InnerProductDecoder, MLP

EPS = 1e-15


class GITPrePromptModel(GFMPrePromptModelBase):
    """
    GIT pretraining model with EMA target encoder and three self-supervised losses.

    Parameters
    ----------
    encoder: GITEncoder
        Online encoder (pygfm backbone + GIT pooling).
    feat_decoder: nn.Module
        Maps hidden_dim → input_dim for feature reconstruction.
    topo_decoder: InnerProductDecoder
        Inner-product decoder for topology reconstruction.
    """

    gfm_family: ClassVar[str] = "git"

    def __init__(self, encoder, feat_decoder, topo_decoder):
        super().__init__()
        self.encoder = encoder
        self.feat_decoder = feat_decoder
        self.topo_decoder = topo_decoder

        self.sem_encoder = deepcopy(self.encoder)
        self.sem_decoder = MLP(
            self.encoder.hidden_dim, self.encoder.hidden_dim,
            self.encoder.hidden_dim, 2, 0.5,
        )

        self.to(self.device)

    # ── Persistence ──────────────────────────────────────────────

    def save_encoder(self, path):
        self.encoder.save(path)

    # ── EMA update ───────────────────────────────────────────────

    def ema_update_sem_encoder(self, decay=0.99):
        for param_q, param_k in zip(
            self.encoder.parameters(), self.sem_encoder.parameters()
        ):
            param_k.data.mul_(decay).add_(param_q.data, alpha=1 - decay)

    # ── Individual losses ────────────────────────────────────────

    def feat_recon_loss(self, z, x, bs=None):
        z = self.feat_decoder(z)
        return F.mse_loss(z[:bs], x[:bs])

    def topo_recon_loss(self, z, pos_edge_index, neg_edge_index=None, ratio=1.0):
        if ratio == 0.0:
            return torch.tensor(0.0, device=z.device)

        if ratio != 1.0:
            num_pos_edges = int(pos_edge_index.size(1) * ratio)
            num_pos_edges = max(num_pos_edges, 1)
            perm = torch.randperm(pos_edge_index.size(1), device=z.device)
            perm = perm[:num_pos_edges]
            pos_edge_index = pos_edge_index[:, perm]

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))

        pos_loss = -torch.log(
            self.topo_decoder(z, pos_edge_index, sigmoid=True) + EPS
        ).mean()
        neg_loss = -torch.log(
            1 - self.topo_decoder(z, neg_edge_index, sigmoid=True) + EPS
        ).mean()

        return pos_loss + neg_loss

    def sem_recon_loss(self, z1, z2, eta=1.0, bs=None):
        h1 = self.sem_decoder(z1)
        h2 = self.sem_decoder(z2)

        z1_n = F.normalize(z1[:bs], dim=-1, p=2).detach()
        z2_n = F.normalize(z2[:bs], dim=-1, p=2).detach()
        h1_n = F.normalize(h1[:bs], dim=-1, p=2)
        h2_n = F.normalize(h2[:bs], dim=-1, p=2)

        loss = (
            (1 - (z1_n * h2_n).sum(dim=-1)).pow(eta)
            + (1 - (z2_n * h1_n).sum(dim=-1)).pow(eta)
        ) / 2
        return loss.mean()

    # ── Forward ──────────────────────────────────────────────────

    def forward(self, x, edge_index, aug1, aug2, bs=None, params=None):
        """
        Parameters
        ----------
        x : Tensor [N, D]
            Original node features.
        edge_index : Tensor [2, E]
            Original edge index.
        aug1 : tuple (x1, edge_index1)
            First augmented view (masked features + dropped edges).
        aug2 : tuple (x2, edge_index2)
            Second augmented view.
        bs : int
            Number of original (non-task) nodes; the remainder are virtual task nodes.
        params : dict
            Hyperparameters (multitask, feat_lambda, topo_lambda, sem_lambda,
            topo_recon_ratio, align_reg_lambda).

        Returns
        -------
        dict with keys: loss, feat_loss, topo_loss, sem_loss, align_reg
        """
        if params is None:
            params = {}

        x1, edge_index1 = aug1
        x2, edge_index2 = aug2

        z1 = self.encoder.encode(x1, edge_index1)
        z2 = self.encoder.encode(x2, edge_index2)

        z1 = self.encoder.pooling(z1, edge_index1)
        z2 = self.encoder.pooling(z2, edge_index2)

        sem_loss = self.sem_recon_loss(z1, z2, eta=1.0, bs=bs)

        device = z1.device
        feat_loss = torch.tensor(0.0, device=device)
        topo_loss = torch.tensor(0.0, device=device)
        align_reg = torch.tensor(0.0, device=device)

        if params.get('multitask', False):
            feat_loss = (
                self.feat_recon_loss(z1, x, bs=bs)
                + self.feat_recon_loss(z2, x, bs=bs)
            ) / 2
            topo_loss = (
                self.topo_recon_loss(
                    z1, edge_index1, ratio=params.get("topo_recon_ratio", 0.1)
                )
                + self.topo_recon_loss(
                    z2, edge_index2, ratio=params.get("topo_recon_ratio", 0.1)
                )
            ) / 2

            if not params.get('pareto', False):
                feat_loss = feat_loss * params.get('feat_lambda', 1)
                topo_loss = topo_loss * params.get('topo_lambda', 1)
                sem_loss = sem_loss * params.get('sem_lambda', 1)

        if params.get('align_reg_lambda', 0) > 0:
            z_mean = z1.mean(0)
            align_reg = (
                F.kl_div(
                    z1.log_softmax(dim=-1),
                    z_mean.softmax(dim=-1),
                    reduction="batchmean",
                )
                * params['align_reg_lambda']
            )

        total = feat_loss + topo_loss + sem_loss + align_reg

        return {
            'loss': total,
            'feat_loss': feat_loss,
            'topo_loss': topo_loss,
            'sem_loss': sem_loss,
            'align_reg': align_reg,
        }

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return pooled node embeddings (for downstream use)."""
        z = self.encoder.encode(x.to(self.device), edge_index.to(self.device))
        return self.encoder.pooling(z, edge_index.to(self.device))
