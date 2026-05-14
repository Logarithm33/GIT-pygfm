import numpy as np
import torch
from torchmetrics import Accuracy
from sklearn.metrics import roc_auc_score

task2metric = {'node': 'acc', 'edge': 'acc', 'graph': 'auc', 'link_pred': 'auc'}


def evaluate(pred, y, mask=None, params=None):
    metric = task2metric[params['task']]

    if metric == 'acc':
        return eval_acc(pred, y, mask) * 100
    elif metric == 'auc':
        return eval_auc(pred, y) * 100
    else:
        raise ValueError(f"Metric {metric} is not supported.")


def eval_acc(y_pred, y_true, mask):
    device = y_pred.device
    num_classes = y_pred.size(1)

    evaluator = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    if mask is not None:
        return evaluator(y_pred[mask], y_true[mask]).item()
    else:
        return evaluator(y_pred, y_true).item()


def eval_auc(y_pred, y_true):
    ndim = y_true.ndim
    if ndim == 1:
        y_pred = y_pred.view(-1, 1)
        y_true = y_true.view(-1, 1)
    elif ndim == 2:
        pass

    rocauc_list = []
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            rocauc_list.append(roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i]))

    if len(rocauc_list) == 0:
        raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list) / len(rocauc_list)
