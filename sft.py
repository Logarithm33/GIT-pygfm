"""
GIT Supervised Fine-Tuning (SFT) entry point.

Loads a pretrained encoder and fine-tunes it on a single domain with MSE
alignment loss between pooled node/edge/graph representations and the
corresponding class text embeddings.
"""

import os
import os.path as osp
import warnings

import torch
import torch.nn as nn
from torch.optim import AdamW

from pygfm.public.utils import set_seed

from data.finetune_data import get_data
from data.pretrain_data import domain2task, dataset2domain
from model.encoder import GITEncoder
from utils.utils import load_params, check_path
from utils.args import get_args_sft
from utils.loader import get_sft_loader

from task.node import sft_node
from task.edge import sft_edge
from task.graph import sft_graph

try:
    import wandb
except ImportError:
    wandb = None

warnings.filterwarnings("ignore")


def get_sft(params):
    task = params["task"]
    if task == "node":
        return sft_node
    elif task == "edge":
        return sft_edge
    elif task == "graph":
        return sft_graph
    else:
        raise ValueError("Invalid Task")


get_loader = get_sft_loader


def run(params):
    params["activation"] = nn.ReLU if params["activation"] == "relu" else nn.LeakyReLU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    data = get_data(params)
    sft = get_sft(params)

    if params["bs"] != 0:
        data = get_loader(data, params)

    sft_model = GITEncoder(
        input_dim=params["input_dim"],
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        backbone=params["backbone"],
        normalize=params["normalize"],
        dropout=params["dropout"],
        activation="relu" if params["activation"] == nn.ReLU else "leaky_relu",
    )

    # Load pretrained encoder if available
    if params["pretrain_dataset"] != 'na':
        template = "lr_{}_hidden_{}_layer_{}_backbone_{}_fp_{}_ep_{}_alignreg_{}_pt_data_{}"
        path = osp.join(
            params['pt_model_path'],
            template.format(
                params['pt_lr'], params['hidden_dim'], params['num_layers'],
                params['backbone'], params['pt_feat_p'], params['pt_edge_p'],
                params['pt_align_reg_lambda'], params['pretrain_dataset'],
            ),
        )
        sft_model = load_params(
            sft_model, osp.join(path, f'encoder_{params["pt_epochs"]}.pt'),
        )
        print("Loaded the pretrained encoder model from {}".format(path))

    sft_model = sft_model.to(device)

    optimizer = AdamW(sft_model.parameters(), lr=params["lr"],
                      weight_decay=params["decay"])

    for epoch in range(1, params['epochs'] + 1):
        sft_loss = sft(model=sft_model, data=data, optimizer=optimizer)

        if wandb is not None:
            wandb.log({'loss/sft_loss': sft_loss})

        if params.get('save', False):
            if epoch % 5 == 0:
                dir_template = "pt_lr_{}_hidden_{}_layer_{}_backbone_{}_fp_{}_ep_{}_alignreg_{}_pt_data_{}_pt_epochs_{}"
                template = "sft_lr_{}_sft_data_{}"
                path = osp.join(
                    params['sft_model_path'],
                    dir_template.format(
                        params['pt_lr'], params['hidden_dim'], params['num_layers'],
                        params['backbone'], params['pt_feat_p'], params['pt_edge_p'],
                        params['pt_align_reg_lambda'], params['pretrain_dataset'],
                        params['pt_epochs'],
                    ),
                    template.format(params['lr'], params['dataset']),
                )
                check_path(path)
                print("Save the instruction fine-tuned model at Epoch {}".format(epoch))
                sft_model.save(osp.join(path, f"encoder_{epoch}.pt"))

    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    params = get_args_sft()
    params['data_path'] = osp.join(os.path.dirname(__file__), '..', 'cache_data')
    params['pt_model_path'] = osp.join(os.path.dirname(__file__), 'model', 'pretrain_model')
    params['sft_model_path'] = osp.join(os.path.dirname(__file__), 'model', 'sft_model')

    dataset = params["dataset"]
    task = domain2task[dataset2domain[dataset]]
    params['task'] = task
    if task == "graph":
        if params['bs'] == 0:
            params['bs'] = 4096

    if params['dataset'] in ['chempcba', 'chemhiv']:
        params['epochs'] = 100

    if params['dataset'] == 'products':
        params['bs'] = 4096

    if wandb is not None:
        wandb.init(
            project="GIT-SFT",
            name="Data:{} | PT-Epoch:{}".format(
                str.upper(params["dataset"]), params["pt_epochs"],
            ),
            mode="disabled" if params["debug"] else "online",
            config=params,
            group=params['group'],
        )
        params = dict(wandb.config)

    print(params)
    run(params)
