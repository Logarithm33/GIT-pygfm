# GIT 复现 (基于 pygfm)

复现 ICML 2025 论文 **"Towards Graph Foundation Models: Learning Generalities Across Graphs via Task-Trees"** 的 GIT 模型，使用 `pygfm` 库作为 GNN backbone 和模型基类。

## 架构概览

三阶段训练范式：

```
pretrain.py ──> sft.py ──> finetune.py
     │              │              │
     ▼              ▼              ▼
GITPrePromptModel  GITEncoder  GITDownPrompt*Model
(GFMPrePrompt-   (SFT on a     (GFMDownPrompt*-
 ModelBase)       single domain) ModelBase)
     │
     ├── feat_recon_loss (MSE)
     ├── topo_recon_loss (InnerProduct + BCE)
     ├── sem_recon_loss  (cross-view cosine)
     └── align_reg       (KL divergence)
```

## pygfm 集成汇总

| pygfm 模块 | 使用位置 | 替代了 GIT 原版什么 |
|-----------|---------|-------------------|
| `public.backbone_models.GraphSAGEEncoderSparse` | `model/encoder.py` | 自定义 `MySAGEConv` |
| `public.backbone_models.GCNEncoderSparse` | `model/encoder.py` | PyG GCNConv (可选) |
| `public.backbone_models.GATEncoderSparse` | `model/encoder.py` | PyG GATConv (可选) |
| `public.model_bases.GFMPrePromptModelBase` | `model/git_pretrain.py` | 裸 `nn.Module` |
| `public.model_bases.GFMDownPromptNodeModelBase` | `model/git_downstream.py` | 裸 `nn.Module` |
| `public.model_bases.GFMDownPromptGraphModelBase` | `model/git_downstream.py` | 裸 `nn.Module` |
| `public.utils.set_seed` | `pretrain.py`, `finetune.py` | `seed_everything()` |

**pygfm 未覆盖的 GIT 特有组件**（保留原版实现）：

| 组件 | 说明 |
|------|------|
| `VirtualNodeAugmentor` | GIT 核心 Task-Tree 构建，pygfm 无对应概念 |
| `NonParamPooling` | 无参消息传递池化，pygfm 无替代 |
| `InnerProductDecoder` | 内积解码器，pygfm 无替代 |
| `proto_classify` | 原型分类（few-shot / zero-shot），pygfm 无替代 |
| `fast_aug` | 未使用：pygfm 是 column-wise 特征掩码，GIT 需 node-wise |

## 文件清单

| 文件 | 来源 | 与原版差异 |
|------|------|-----------|
| `.gitignore` | 新编 | — |
| `__init__.py` | 新编 | — |
| `config/*.yaml` | 复制 (4个) | 无修改 |
| `utils/args.py` | 搬运 | 无修改 |
| `utils/eval.py` | 搬运 | 删除未使用注释代码 (~100行) |
| `utils/logger.py` | 搬运 | 无修改 |
| `utils/early_stop.py` | 混合 | 新增 pygfm 集成声明 |
| `utils/utils.py` | 搬运 | 删除 3 个暂未使用的函数 |
| `utils/loader.py` | 搬运 | 提取 `clean_data` 辅助函数 |
| `utils/split.py` | 搬运 | 修复 `np.where` 多值警告 |
| `model/encoder.py` | 混合 | `MySAGEConv` → pygfm backbone；MLP 拆分到 decoders |
| `model/decoders.py` | 混合 | `InnerProductDecoder` + `MLP` 从 encoder 拆分 |
| `model/git_pretrain.py` | 改写 | `nn.Module` → `GFMPrePromptModelBase`；新增 `embed()` |
| `model/git_downstream.py` | 改写 | `nn.Module` → `GFMDownPrompt*Base`；`scatter_mean` → for-loop |
| `data/finetune_data.py` | 搬运 | `int()` → `.item()`；pandas 惰性导入 |
| `data/ofa_dataset.py` | 搬运 | 删除未使用 import |
| `data/pretrain_data.py` | 搬运 | `groups.max()` → `.item()`；新增 docstring |
| `task/node.py` | 搬运 | 无逻辑修改 |
| `task/edge.py` | 搬运 | 无逻辑修改 |
| `task/link_pred.py` | 搬运 | 无逻辑修改 |
| `task/graph.py` | 搬运 | 无逻辑修改 |
| `pretrain.py` | 改写 | `Encoder` → `GITEncoder`；`PretrainModel` → `GITPrePromptModel`；wandb 可选 |
| `sft.py` | 改写 | `Encoder` → `GITEncoder`；wandb 可选；修复原版缩进错误 |
| `finetune.py` | 改写 | `TaskModel` → `GITDownPrompt*Model`；wandb 可选；yaml 惰性导入 |
| `tests/*.py` | 新编 (8个) | — |

### 统计

| 类别 | 数量 | 文件 |
|------|------|------|
| 复制 | 6 | 4 YAML + args + logger |
| 搬运 | 9 | eval, utils, loader, split, finetune_data, ofa_dataset, pretrain_data, task×4 |
| 改写 | 6 | encoder, decoders, git_pretrain, git_downstream, pretrain, sft, finetune |
| 新编 | 13 | .gitignore, README, 5 `__init__.py`, 8 tests |

## 项目结构

```
main/
├── README.md
├── __init__.py
├── .gitignore
├── pretrain.py
├── sft.py
├── finetune.py
├── utils/
│   ├── __init__.py
│   ├── args.py
│   ├── eval.py
│   ├── logger.py
│   ├── early_stop.py
│   ├── utils.py
│   ├── loader.py
│   └── split.py
├── model/
│   ├── __init__.py
│   ├── encoder.py
│   ├── decoders.py
│   ├── git_pretrain.py
│   └── git_downstream.py
├── data/
│   ├── __init__.py
│   ├── finetune_data.py
│   ├── ofa_dataset.py
│   └── pretrain_data.py
├── task/
│   ├── __init__.py
│   ├── node.py
│   ├── edge.py
│   ├── link_pred.py
│   └── graph.py
├── config/
│   ├── base.yaml
│   ├── base_zero_shot.yaml
│   ├── zero_shot.yaml
│   └── in_context.yaml
└── tests/
    ├── test_step1_utils.py
    ├── test_step2_model.py
    ├── test_step3_data.py
    ├── test_step4_pretrain_data.py
    ├── test_step5_pretrain_model.py
    ├── test_step7_downstream_model.py
    ├── test_step8_task_functions.py
    ├── test_step9_sft_entry.py
    └── test_step10_finetune_entry.py
```

## 测试覆盖

| 测试文件 | 覆盖对象 | 测试数 |
|---------|---------|--------|
| `test_step1_utils.py` | args, eval, logger, early_stop, utils | 19 |
| `test_step2_model.py` | GITEncoder, NonParamPooling, InnerProductDecoder, MLP | 20 |
| `test_step3_data.py` | registries, OFA base, loaders, temporal helpers | 13 |
| `test_step4_pretrain_data.py` | VirtualNodeAugmentor, preprocess, unified_data | 18 |
| `test_step5_pretrain_model.py` | GITPrePromptModel, EMA, 3 losses, gradient flow | 13 |
| `test_step7_downstream_model.py` | DownPromptNode/Graph, l2norm, get_prototypes, proto_classify | 18 |
| `test_step8_task_functions.py` | sft/ft/eval for node/edge/link_pred/graph | 18 |
| `test_step9_sft_entry.py` | sft.py module, get_sft dispatch, imports | 5 |
| `test_step10_finetune_entry.py` | finetune.py module, get_ft/get_eval dispatch | 9 |
| **合计** | | **133** |
