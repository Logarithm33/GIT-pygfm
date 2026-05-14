import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class InnerProductDecoder(nn.Module):
    """
    Inner-product decoder for link reconstruction.

    From ``"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>``_.
    Computes :math:`\\sigma(z_i^\\top z_j)` for node pairs.

    Parameters
    ----------
    hidden_dim: int, optional
        If provided, applies a linear projection before inner product.
    output_dim: int, optional
        Output dimension of the projection.
    """

    def __init__(self, hidden_dim: int = None, output_dim: int = None):
        super().__init__()
        self.proj_z = hidden_dim is not None
        if self.proj_z:
            self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: Tensor, edge_index: Tensor, sigmoid: bool = True) -> Tensor:
        z = self.lin(z) if self.proj_z else z
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z: Tensor, sigmoid: bool = True) -> Tensor:
        z = self.lin(z) if self.proj_z else z
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj


class MLP(nn.Module):
    """
    Simple MLP with BatchNorm and optional ReLU ordering.

    Parameters
    ----------
    in_channels: int
    hidden_channels: int
    out_channels: int
    num_layers: int
        Total number of linear layers (>=2).
    dropout: float
    relu_first: bool
        If True, ReLU → BN → Dropout.  Otherwise BN → ReLU → Dropout.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.5,
        relu_first: bool = True,
    ):
        super().__init__()
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.relu_first = relu_first

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.relu_first:
                x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            if not self.relu_first:
                x = F.relu(x, inplace=True)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
