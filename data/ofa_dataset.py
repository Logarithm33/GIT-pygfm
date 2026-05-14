import os
import os.path as osp
from collections.abc import Mapping
from typing import Optional, Callable, Any

import torch
import numpy as np
import torch_geometric as pyg
from torch_geometric.data import InMemoryDataset
from abc import ABC, abstractmethod


def safe_mkdir(path):
    if not osp.exists(path):
        os.mkdir(path)


def pth_safe_save(obj, path):
    if obj is not None:
        torch.save(obj, path)


def pth_safe_load(path):
    if osp.exists(path):
        return torch.load(path)
    return None


class OFAPygDataset(InMemoryDataset, ABC):
    """
    Base dataset class for OFA-format datasets. Handles dataset loading,
    text-to-feature transformation, and storage of side data.

    Subclasses must implement ``gen_data``, ``add_text_emb``, ``get_task_map``,
    and ``get_edge_list``.
    """

    def __init__(
        self,
        name: str,
        root: str = "./cache_data",
        load_text: bool = False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.name = name
        self.root = root
        self.data_dir = osp.join(self.root, self.name)
        super().__init__(self.data_dir, transform, pre_transform)
        safe_mkdir(self.data_dir)

        if load_text:
            self.texts = torch.load(self.processed_paths[1])

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.side_data = pth_safe_load(self.processed_paths[2])

    def data2vec(self, data: list[str]) -> torch.Tensor:
        if self.encoder is None:
            raise NotImplementedError("LLM encoder is not defined")
        if data is None:
            return None
        data = np.nan_to_num(data, nan="")
        embeddings = self.encoder.encode(data).cpu()
        return embeddings

    @property
    def num_classes(self):
        return self.__num_classes__

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["geometric_data_processed.pt", "texts.pkl", "data.pt"]

    def text2feature(self, texts):
        if isinstance(texts, list) and len(texts) == 0:
            return []
        if isinstance(texts[0], str):
            return self.data2vec(texts)
        return [self.text2feature(t) for t in texts]

    @abstractmethod
    def gen_data(self) -> tuple[list[pyg.data.Data], list[list[str]], Any]:
        pass

    @abstractmethod
    def add_text_emb(
        self, data_list, texts_emb: list[torch.Tensor]
    ) -> tuple[pyg.data.Data, Mapping]:
        pass

    def process(self):
        if self.encoder.model is None:
            self.encoder.get_model()
        data_list, texts, side_data = self.gen_data()

        texts_emb = self.text2feature(texts)
        torch.save(texts, self.processed_paths[1])
        if side_data is not None:
            torch.save(side_data, self.processed_paths[2])
        else:
            torch.save("No side data", self.processed_paths[2])
        data, slices = self.add_text_emb(data_list, texts_emb)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    @abstractmethod
    def get_task_map(self) -> dict[str, dict]:
        pass

    @abstractmethod
    def get_edge_list(self, mode="e2e") -> dict[str, list]:
        pass

    def get_prompt_text_feat(self, task_name):
        task_map = self.get_task_map()
        if task_name not in task_map:
            raise NotImplementedError(
                "Task " + task_name + " is not implemented for "
                + self.name + " dataset the implemented tasks are "
                + str(task_map.keys())
            )
        feat_ind = task_map[task_name]
        prompt_feats = {}
        for k in feat_ind:
            prompt_feats[k] = getattr(self.data, feat_ind[k][0])[feat_ind[k][1]]
        return prompt_feats


class MolOFADataset(OFAPygDataset):
    """OFA-format molecule dataset."""

    def gen_data(self):
        pass

    def add_text_emb(self, data_list, text_emb):
        data, slices = self.collate(data_list)
        data.node_embs = text_emb[0]
        data.edge_embs = text_emb[1]
        data.class_node_text_feat = text_emb[2]
        data.prompt_edge_text_feat = text_emb[3]
        data.noi_node_text_feat = text_emb[4]
        return data, slices

    def get(self, index):
        data = super().get(index)
        node_feat = self.data.node_embs[data.x]
        edge_feat = self.data.edge_embs[data.xe]
        data.node_text_feat = node_feat
        data.edge_text_feat = edge_feat
        data.y = data.y.view(1, -1)
        data.x = data.node_text_feat
        data.xe = data.edge_text_feat
        return data

    def get_idx_split(self):
        return self.side_data[0]

    def get_task_map(self):
        return self.side_data[1]

    def get_edge_list(self, mode="e2e"):
        if mode == "e2e_graph":
            return {"f2n": [1, 0], "n2f": [3, 0], "n2c": [2, 0]}
        elif mode == "lr_graph":
            return {"f2n": [1, 0], "n2f": [3, 0]}
