from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn.aggr import MultiAggregation
from torch_geometric.typing import Adj, OptPairTensor, SparseTensor
from torch_geometric.utils import spmm

from pygfm.public.backbone_models import (
    GraphSAGEEncoderSparse,
    GCNEncoderSparse,
    GATEncoderSparse,
)


class NonParamPooling(MessagePassing):
    """Parameter-free mean-aggregation pooling via message passing.

    Equivalent to GIT-main's ``NonParamPooling``. Performs one round of
    mean message passing over the edge index, aggregating neighbour features.
    """
    def __init__(self, aggr="mean"):
        super().__init__(aggr)

    def forward(self, x, edge_index, edge_attr=None):
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        out = self.propagate(edge_index, x=x)
        return out

    def message(self, x_j):
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: OptPairTensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)


class GITEncoder(nn.Module):
    """
    GIT Encoder wrapping pygfm's sparse GNN backbone.

    Uses pygfm ``GraphSAGEEncoderSparse`` / ``GCNEncoderSparse`` / ``GATEncoderSparse``
    for the convolutional layers, and keeps GIT-specific ``NonParamPooling`` + linear
    projection for the task-node readout.

    Parameters
    ----------
    input_dim: int
        Input feature dimension (text embedding dim, e.g. 768).
    hidden_dim: int
        Hidden / output dimension.
    num_layers: int
        Number of GNN convolution layers.
    backbone: str
        One of ``"sage"``, ``"gcn"``, ``"gat"``.
    activation: str
        Activation name (``"relu"``, ``"gelu"``, ``"tanh"``).
    normalize: str
        ``"batch"`` for BatchNorm, ``"none"`` otherwise.
    dropout: float
        Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        backbone: str = "sage",
        activation: str = "relu",
        normalize: str = "batch",
        dropout: float = 0.15,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.backbone = backbone
        self.normalize = normalize

        use_bn = normalize == "batch"

        common = dict(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
            use_batch_norm=use_bn,
        )

        if backbone == "sage":
            self.conv = GraphSAGEEncoderSparse(**common)
        elif backbone == "gcn":
            self.conv = GCNEncoderSparse(**common)
        elif backbone == "gat":
            self.conv = GATEncoderSparse(**common, heads=4)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.mean_aggr = NonParamPooling(aggr="mean")
        self.pooling_lin = nn.Linear(hidden_dim, hidden_dim)

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.conv, 'reset_parameters'):
            self.conv.reset_parameters()
        self.pooling_lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None) -> Tensor:
        z = self.encode(x, edge_index, edge_attr)
        return z

    def encode(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None) -> Tensor:
        """Encode node features through GNN conv layers (no pooling)."""
        z = self.conv(x, edge_index)
        return z

    def pooling(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Task-node pooling: mean-aggregation message passing + linear projection."""
        z = self.mean_aggr(x, edge_index)
        z = self.pooling_lin(z)
        return z

    def encode_graph(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
        pool: str = "mean",
    ) -> Tensor:
        """Encode and global-pool to graph-level embeddings."""
        z = self.encode(x, edge_index)
        if pool == "mean":
            z = global_mean_pool(z, batch)
        elif pool == "sum":
            z = global_add_pool(z, batch)
        elif pool == "max":
            z = global_max_pool(z, batch)
        return z

    def save(self, path: str):
        torch.save(self.state_dict(), path)
