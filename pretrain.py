"""
GIT Pretraining entry point.

Builds task-tree unified data, trains GITPrePromptModel with dual-view
augmentation and multi-loss self-supervision.

Uses pygfm for seeding (set_seed).
"""

import os
import os.path as osp
import random

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.utils import mask_feature, dropout_edge

from pygfm.public.utils import set_seed

from data.pretrain_data import unified_data
from model.encoder import GITEncoder
from model.decoders import InnerProductDecoder
from model.pretrain_model import GITPrePromptModel
from utils.utils import get_scheduler, get_device_from_model, check_path
from utils.args import get_args_pretrain
from utils.loader import get_pt_loader

try:
    import wandb
except ImportError:
    wandb = None

get_loader = get_pt_loader


def pretrain(model, loader, optimizer, scheduler=None, **kwargs):
    """Single-epoch training loop."""
    model.train()
    device = get_device_from_model(model)
    params = kwargs['params']

    for data in loader:
        bs = data.batch_size

        x = data.node_text_feat[data.x].to(device)
        edge_index = data.edge_index.to(device)

        x1, _ = mask_feature(x, p=params["feat_p"])
        edge_index1, _ = dropout_edge(edge_index, p=params["edge_p"],
                                      force_undirected=True)
        x2, _ = mask_feature(x, p=params["feat_p"])
        edge_index2, _ = dropout_edge(edge_index, p=params["edge_p"],
                                      force_undirected=True)

        losses = model(x, edge_index,
                       aug1=(x1, edge_index1),
                       aug2=(x2, edge_index2),
                       bs=bs, params=params)
        loss = losses['loss']

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if scheduler:
            scheduler.step()

        if wandb is not None:
            wandb.log({
                "loss/feat_loss": losses["feat_loss"].item(),
                "loss/topo_loss": losses["topo_loss"].item(),
                "loss/sem_loss": losses["sem_loss"].item(),
                "loss/align_reg": losses["align_reg"].item(),
                "loss/loss": loss.item(),
            })


def run(params):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(params["seed"])
    params["activation"] = nn.ReLU if params["activation"] == "relu" else nn.LeakyReLU

    pretrain_data, task_node_idx_dict = unified_data(params)
    train_nodes = torch.cat(list(task_node_idx_dict.values()))
    if params['train_ratio'] != 1:
        train_nodes = torch.tensor(
            random.sample(train_nodes.tolist(),
                          int(len(train_nodes) * params['train_ratio']))
        )
    print("Number of training nodes is {}".format(len(train_nodes)))

    encoder = GITEncoder(
        input_dim=params["input_dim"],
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        backbone=params["backbone"],
        activation="relu" if params["activation"] == nn.ReLU else "leaky_relu",
        normalize=params["normalize"],
        dropout=params["dropout"],
    )
    feat_decoder = nn.Linear(params["hidden_dim"], params["input_dim"])
    topo_decoder = InnerProductDecoder(
        hidden_dim=params["hidden_dim"], output_dim=params["hidden_dim"],
    )
    pretrain_model = GITPrePromptModel(
        encoder=encoder, feat_decoder=feat_decoder, topo_decoder=topo_decoder,
    ).to(device)

    optimizer = AdamW(
        pretrain_model.parameters(),
        lr=params["lr"], weight_decay=params["decay"],
    )
    scheduler = get_scheduler(optimizer, params["use_schedular"], params["epochs"])

    for i in range(1, params["epochs"] + 1):
        loader = get_loader(pretrain_data, train_nodes, params)
        print("Number of mini-batches is {} at epoch {}.".format(len(loader), i))

        pretrain(model=pretrain_model, loader=loader, optimizer=optimizer,
                 scheduler=scheduler, params=params)

        template = "lr_{}_hidden_{}_layer_{}_backbone_{}_fp_{}_ep_{}_alignreg_{}_pt_data_{}"
        if params['train_ratio'] != 1:
            template += "_{}".format(params['train_ratio'])

        save_path = osp.join(
            params['model_path'],
            template.format(
                params["lr"], params["hidden_dim"], params['num_layers'],
                params["backbone"], params["feat_p"], params["edge_p"],
                params["align_reg_lambda"], params["pretrain_dataset"],
            ),
        )
        check_path(save_path)
        pretrain_model.save_encoder(osp.join(save_path, f"encoder_{i}.pt"))
        print("Save the model at epoch {}".format(i))

    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    params = get_args_pretrain()
    params['data_path'] = osp.join(os.path.dirname(__file__), '..', 'cache_data')
    params['model_path'] = osp.join(os.path.dirname(__file__), 'model', 'pretrain_model')

    if wandb is not None:
        wandb.init(
            project="GIT-Pretrain",
            name="LR:{} | Layers:{} | Fan:{}".format(
                params["lr"], params["num_layers"], params["fanout"],
            ),
            mode="disabled" if params["debug"] else "online",
            group=params['group'],
            config=params,
        )

    run(params)
