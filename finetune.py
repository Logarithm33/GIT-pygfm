"""
GIT Downstream Finetuning entry point.

Supports four settings:
  - ``base`` — standard full-data fine-tuning
  - ``few_shot`` — N-way K-shot episodic learning
  - ``zero_shot`` — zero-shot using class text embeddings as prototypes
  - ``in_context`` — in-context learning with support-set prototypes
  - ``base_zero_shot`` — zero-shot without any training

Uses ``pygfm.set_seed`` for reproducibility.
"""

import os
import os.path as osp
from copy import deepcopy

import torch
import torch.nn as nn
from torch.optim import AdamW

from pygfm.public.utils import set_seed

from data.finetune_data import get_data
from data.pretrain_data import domain2task, dataset2domain
from model.encoder import GITEncoder
from model.finetune_model import GITDownPromptNodeModel, GITDownPromptGraphModel
from utils.utils import load_params, mask2idx, check_path
from utils.args import get_args_finetune
from utils.early_stop import EarlyStopping
from utils.logger import Logger
from utils.split import get_split
from utils.loader import get_ft_loader

from task.node import ft_node, eval_node, eval_node_few_shot
from task.edge import ft_edge, eval_edge, eval_edge_few_show
from task.link_pred import ft_link_pred, eval_link_pred
from task.graph import ft_graph, eval_graph

try:
    import wandb
except ImportError:
    wandb = None

import warnings
warnings.filterwarnings("ignore")


# ── Dispatch ────────────────────────────────────────────────────────

def get_ft(params):
    task = params["task"]
    if task == "node":
        return ft_node
    elif task == "edge":
        return ft_edge
    elif task == "link_pred":
        return ft_link_pred
    elif task == "graph":
        return ft_graph
    else:
        raise ValueError("Does not support the task in finetuning.")


def get_eval(params):
    setting = params["setting"]
    task = params["task"]

    if task == "node":
        if setting in ['base', 'base_zero_shot']:
            return eval_node
        elif setting in ['few_shot', 'zero_shot', 'in_context']:
            return eval_node_few_shot
    elif task == "edge":
        if setting in ['base', 'base_zero_shot']:
            return eval_edge
        elif setting in ['few_shot', 'zero_shot', 'in_context']:
            return eval_edge_few_show
    elif task == "link_pred":
        if setting in ['base']:
            return eval_link_pred
        elif setting in ['base_zero_shot', 'few_shot', 'zero_shot', 'in_context']:
            raise ValueError("Not support the setting yet in evaluation.")
    elif task == "graph":
        return eval_graph
    else:
        raise ValueError("Does not support the task in evaluation.")


get_loader = get_ft_loader


# ── Main ────────────────────────────────────────────────────────────

def run(params):
    params["activation"] = nn.ReLU if params["activation"] == "relu" else nn.LeakyReLU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    graph = get_data(params)
    splits = get_split(graph, params)
    finetune = get_ft(params)
    evaluate = get_eval(params)

    # Auto-detect input_dim from data if using default (LLM features = 768, raw features vary)
    actual_dim = graph.node_text_feat.shape[1]
    if params["input_dim"] == 768 and actual_dim != 768:
        print(f"Auto-detected input_dim: {actual_dim} (was {params['input_dim']})")
        params["input_dim"] = actual_dim

    encoder = GITEncoder(
        input_dim=params["input_dim"],
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        backbone=params["backbone"],
        normalize=params["normalize"],
        dropout=params["dropout"],
        activation="relu" if params["activation"] == nn.ReLU else "leaky_relu",
    )

    if params["pt_data"] != 'na':
        if params['sft_data'] == 'na':
            template = "lr_{}_hidden_{}_layer_{}_backbone_{}_fp_{}_ep_{}_alignreg_{}_pt_data_{}"
            if params['train_ratio'] != 1.0:
                template += "_{}".format(params['train_ratio'])
            base_path = params['pt_model_path']
            path = osp.join(
                base_path,
                template.format(
                    params['pt_lr'], params['hidden_dim'], params['num_layers'],
                    params['backbone'], params['pt_feat_p'], params['pt_edge_p'],
                    params['pt_align_reg_lambda'], params['pt_data'],
                ),
                f"encoder_{params['pt_epochs']}.pt",
            )
        else:
            dir_template = "pt_lr_{}_hidden_{}_layer_{}_backbone_{}_fp_{}_ep_{}_alignreg_{}_pt_data_{}_pt_epochs_{}"
            template = "sft_lr_{}_sft_data_{}"
            path = osp.join(
                params['sft_model_path'],
                dir_template.format(
                    params['pt_lr'], params['hidden_dim'], params['num_layers'],
                    params['backbone'], params['pt_feat_p'], params['pt_edge_p'],
                    params['pt_align_reg_lambda'], params['pt_data'],
                    params['pt_epochs'],
                ),
                template.format(params['sft_lr'], params['sft_data']),
                f"encoder_{params['sft_epochs']}.pt",
            )
        check_path(path)
        encoder = load_params(encoder, path)
        print("Load the pretrained model from {}".format(path))

    # Select downstream model based on task
    num_classes = graph.num_classes
    if params['task'] in ('node', 'edge', 'link_pred'):
        model = GITDownPromptNodeModel(encoder, num_classes=num_classes)
    else:
        model = GITDownPromptGraphModel(encoder, num_classes=num_classes)
    model = model.to(device)

    logger = Logger()

    for idx, split in enumerate(splits):
        set_seed(idx)

        if params["bs"] == 0:
            data = deepcopy(graph)
            if params['task'] == 'link_pred':
                data = split(data)
        else:
            data = get_loader(graph, split, params)

        task_model = deepcopy(model)
        optimizer = AdamW(task_model.parameters(), lr=params["lr"],
                          weight_decay=params["decay"])
        stopper = EarlyStopping(patience=params["early_stop"])

        for epoch in range(1, params["epochs"] + 1):
            loss = finetune(model=task_model, data=data, split=split,
                            optimizer=optimizer, params=params)
            result = evaluate(model=task_model, data=data, split=split,
                              params=params)

            is_stop = stopper(result)
            logger.log(idx, epoch, loss, result)
            if is_stop:
                print("Early Stopping at Epoch:", epoch)
                break

            if wandb is not None:
                wandb.log({
                    "train/loss_train": loss,
                    "train/train": result['train'],
                    "train/val": result['val'],
                    "train/test": result['test'],
                    "train/metric": result['metric'],
                })

        single_best = logger.get_single_best(idx)
        if wandb is not None:
            wandb.log({
                "best/train": single_best["train"],
                "best/val": single_best["val"],
                "best/test": single_best["test"],
            })

    best = logger.get_best()
    if wandb is not None:
        wandb.log({
            "final/train": "{:.2f} ± {:.2f}".format(
                best['train']['mean'], best['train']['std']),
            "final/val": "{:.2f} ± {:.2f}".format(
                best['val']['mean'], best['val']['std']),
            "final/test": "{:.2f} ± {:.2f}".format(
                best['test']['mean'], best['test']['std']),
            "final/train_mean": best['train']['mean'],
            "final/val_mean": best['val']['mean'],
            "final/test_mean": best['test']['mean'],
            "final/train_std": best['train']['std'],
            "final/val_std": best['val']['std'],
            "final/test_std": best['test']['std'],
        })
        wandb.log({'meta/run': logger.get_run_raw(),
                   'meta/best': logger.get_best_raw()})
        wandb.finish()


def main():
    params = get_args_finetune()
    params['data_path'] = osp.join(os.path.dirname(__file__), '..', 'cache_data')
    params['pt_model_path'] = osp.join(os.path.dirname(__file__), 'model', 'pretrain_model')
    params['sft_model_path'] = osp.join(os.path.dirname(__file__), 'model', 'sft_model')
    params['ft_model_path'] = osp.join(os.path.dirname(__file__), 'model', 'finetune_model')

    dataset = params["dataset"]
    default_task = domain2task[dataset2domain[dataset]]
    if params['task'] is None:
        params['task'] = default_task
    task = params['task']
    if task == "graph":
        if params['bs'] == 0:
            params['bs'] = 1024

    if params["use_params"]:
        try:
            import yaml
            with open(f"config/{params['setting']}.yaml", "r") as f:
                default_params = yaml.safe_load(f)
                params.update(default_params['base'])
                params.update(default_params[task][dataset])
        except (ImportError, FileNotFoundError, KeyError) as e:
            print(f"Warning: could not load config params: {e}")

    if params["setting"] in ["zero_shot", "in_context"]:
        params["n_task"] = 500
        params["epochs"] = 1
    elif params['setting'] in ['base_zero_shot']:
        params['epochs'] = 1
        params['repeat'] = 1

    if params['dataset'] == 'products':
        params['bs'] = 1024

    if params['dataset'] == 'chempcba':
        params['n_task'] = 50

    if wandb is not None:
        tags = [params.get('task', 'none'), params['setting']]
        wandb.init(
            project="GIT-Finetune",
            name="Data:{} | SFT:{} | PT-Epoch:{}".format(
                params["dataset"], params["sft_data"], params["pt_epochs"]),
            config=params,
            mode="disabled" if params["debug"] else "online",
            group=params.get('group', 'base'),
            tags=tags,
        )
        params = dict(wandb.config)
    print(params)

    run(params)


if __name__ == "__main__":
    main()
