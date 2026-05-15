# GIT 复现 (基于 pygfm)

复现 ICML 2025 论文 **"Towards Graph Foundation Models: Learning Generalities Across Graphs via Task-Trees"** 的 GIT 模型，使用 `pygfm` 库作为基础设施。

## 与 GIT 原版的异同

| 组件 | GIT 原版 | 本版 | 说明 |
|------|---------|------|------|
| 模型基类 | 裸 `nn.Module` | `GFMPrePromptModelBase` / `GFMDownPrompt*Base` | pygfm 提供 |
| GNN Conv | 自定义 `MySAGEConv` | `GraphSAGEEncoderSparse` | pygfm 标准 PyG SAGEConv |
| 无参池化 | `NonParamPooling` | `NonParamPooling` (保留) | pygfm 无此组件 |
| 内积解码器 | `InnerProductDecoder` | `InnerProductDecoder` (保留) | pygfm 无此组件 |
| 工具函数 | 手写 `seed_everything` | `pygfm.public.utils.set_seed` | pygfm 提供 |
| 早停 | `EarlyStopping` 类 | 同上 + 内部可选 pygfm | 两者兼用 |
| 评估 | `torchmetrics` + `sklearn` | 同原版 | pygfm 无封装 |
| 数据 | VirtualNodeAugmentor 等 | 保留原版逻辑 | pygfm 无此概念 |

## pygfm 使用情况

### Step 2 — Encoder & Decoder

| pygfm 模块 | 用途 | 替代了 GIT 原版什么 |
|-----------|------|-------------------|
| `pygfm.public.backbone_models.GraphSAGEEncoderSparse` | SAGE 卷积层 (默认 backbone) | 原版自定义 `MySAGEConv` |
| `pygfm.public.backbone_models.GCNEncoderSparse` | GCN 卷积层 (可选 backbone) | PyG GCNConv |
| `pygfm.public.backbone_models.GATEncoderSparse` | GAT 卷积层 (可选 backbone) | PyG GATConv |

**未使用 pygfm 的组件**：
- `NonParamPooling` — pygfm 不提供，保留自 GIT 原版
- `InnerProductDecoder` — pygfm 不提供内积解码器，保留自 GIT 原版
- `MLP` — pygfm 的 `FeatureEngineeringMLP` 过于复杂（残差+多激活），保留原版简洁 MLP
- `pooling_lin` — GIT 特有设计，`nn.Linear(hidden_dim, hidden_dim)`

### Step 3 — Data Layer (Downstream)

| pygfm 模块 | 用途 |
|-----------|------|
| _本步未直接使用 pygfm_ | 数据加载依赖 PyTorch Geometric，GIT 数据格式 (OFA) 为项目特有 |

**未使用 pygfm 的原因**：
- pygfm 的 `load_all_datasets` 仅支持 Planetoid/Amazon，不覆盖 GIT 的 30+ 数据集
- pygfm 的 `PyGGraph` 是 DGL 兼容包装，GIT 直接使用 PyG Data
- OFA 数据集格式（去重文本嵌入+按索引查找）为项目特有逻辑
- 可选用 pygfm `BertTextEncoder` 做备用文本编码，但缓存数据已含预计算嵌入

### Step 10 — Finetune Entry Point

| pygfm 模块 | 用途 | 替代了 GIT 原版什么 |
|-----------|------|-------------------|
| `pygfm.public.utils.set_seed` | 全局随机种子 | 原版 `seed_everything()` |

**与 GIT 原版异同**：
- `Encoder` → `GITEncoder`、`TaskModel` → `GITDownPromptNodeModel` / `GITDownPromptGraphModel`（按 task 自动选择）
- `seed_everything()` → `pygfm.public.utils.set_seed()`
- `wandb` 改为可选导入
- `yaml` 改为惰性导入（`use_params` 路径），缺失时降级运行
- `check_path` 调用修正位置（原版在路径构造前调用，当模型不存在时路径未创建会导致错误）
- 移除未使用的 import（`numpy`, `shutil`, `F`, `mask_feature`, `negative_sampling`, `dropout_adj` 等）
- 4 种 setting 全支持：base / few_shot / zero_shot / in_context / base_zero_shot

### Step 9 — SFT Entry Point

| pygfm 模块 | 用途 | 替代了 GIT 原版什么 |
|-----------|------|-------------------|
| `pygfm.public.utils.set_seed` | 全局随机种子 | 原版 `seed_everything()` |

**与 GIT 原版异同**：
- `Encoder` → `GITEncoder`（使用 pygfm backbone）
- `wandb` 改为可选导入
- 修复原版 `if task == "graph": params['bs'] = 4096` 的缩进语法错误
- 移除原版 `use_params` 加载 `sft_param.yaml`（该文件不存在）
- 移除未使用的 import（`numpy`, `yaml`, `deepcopy`, `negative_sampling` 等）
- `pretrain_dataset == 'na'` 时跳过预训练权重加载，使用随机初始化

### Step 8 — Task Training / Eval Functions

| pygfm 模块 | 用途 |
|-----------|------|
| _本步未直接使用 pygfm_ | task 函数为 GIT 训练/评估逻辑，与模型框架无关 |

**与 GIT 原版异同**：
- 四个文件 (`node.py` / `edge.py` / `link_pred.py` / `graph.py`) 从 GIT-main 搬运，逻辑完全一致
- `task/edge.py` 的 `temporal_datasets` 导入路径从 `data.pretrain_data` 获取（已在 pretrain_data 中从 finetune_data 导入）
- SFT 函数 (`sft_node` / `sft_edge` / `sft_graph`) 使用 MSE loss 对齐类别文本嵌入
- Fine-tune 函数 (`ft_node` / `ft_edge` / `ft_link_pred` / `ft_graph`) 使用 CE/BCE/Multi-task BCE
- Eval 函数支持 base / few_shot / zero_shot / in_context 四种范式

### Step 7 — Downstream Models (GITDownPrompt)

| pygfm 模块 | 用途 | 替代了 GIT 原版什么 |
|-----------|------|-------------------|
| `pygfm.public.model_bases.GFMDownPromptNodeModelBase` | 节点/边/链接下游模型基类 | 原版 `TaskModel(nn.Module)` |
| `pygfm.public.model_bases.GFMDownPromptGraphModelBase` | 图下游模型基类 | 原版 `TaskModel(nn.Module)` |

**与 GIT 原版异同**：
- 原版单个 `TaskModel(nn.Module)` → 本版两个模型继承不同 pygfm 基类
- `gfm_family = "git"`，`gfm_stage` 分别为 `downprompt_node` / `downprompt_graph`
- 保留 `einops.rearrange`（用户已安装）
- `torch_scatter.scatter_mean` → 纯 PyTorch for-loop（该包需 C++ 编译器，当前环境不可用）
- 保留 `distance_metric`、`l2norm`、`get_prototypes`、`proto_classify` 核心逻辑
- 保留 `classify()`、`encode()`、`encode_graph()`、`pooling_lin()` 接口
- 放弃原版 `compute_multitask_loss`（未使用，且与 task/graph.py 中实现重复）

### Step 6 — Pretrain Entry + Loader + Split

| pygfm 模块 | 用途 | 替代了 GIT 原版什么 |
|-----------|------|-------------------|
| `pygfm.public.utils.set_seed` | 全局随机种子 | 原版 `seed_everything()` |

**未使用 pygfm 的原因**：
- `pygfm.public.utils.fast_aug` 是按特征维零掩（column-wise），而 GIT 原版 `mask_feature` 是节点级零掩（node-wise），语义不同，故保留 `torch_geometric.utils.mask_feature`
- DataLoader 和 split 逻辑使用 PyG 原生接口，pygfm 无替代

**与 GIT 原版异同**：
- `pretrain.py`：
  - `seed_everything()` → `pygfm.public.utils.set_seed()`
  - `Encoder` → `GITEncoder`、`PretrainModel` → `GITPrePromptModel`
  - 增广方式从 `mask_feature/dropout_adj` → `mask_feature/dropout_edge`（适配新版 PyG）
  - `wandb` 改为可选导入，缺失时不崩溃
  - `data_path` 默认指向 `../cache_data`（根目录）
- `loader.py`：搬运自 GIT-main，提取 `clean_data` 辅助函数
- `split.py`：搬运自 GIT-main，`few_shot_split` 中 `np.where` 结果直接传 `random.choice` 修复多值问题

### Step 5 — Pretraining Model (GITPrePromptModel)

| pygfm 模块 | 用途 | 替代了 GIT 原版什么 |
|-----------|------|-------------------|
| `pygfm.public.model_bases.GFMPrePromptModelBase` | 模型基类 | 原版裸 `nn.Module` |
| `pygfm.public.model_bases.GFMModelDescriptor` | 模型描述（family/stage） | 无对应，pygfm 特有 |
| `pygfm.public.utils.loss_func.NodeNodeContrastiveLoss` | （可选）替代 sem_recon_loss | 原版手写余弦相似度 loss |
| `pygfm.public.utils.loss_func.sample_negative_pairs` | （可选）topo_recon 负采样 | 原版用 `torch_geometric.utils.negative_sampling` |

**与 GIT 原版异同**：
- 原版 `PretrainModel(nn.Module)` → 本版 `GITPrePromptModel(GFMPrePromptModelBase)`
- 模型描述信息 `gfm_family="git"`、`describe()` 为 pygfm 特有
- `forward()` 参数化：原版 `forward(graph, aug_g1, aug_g2)` → 本版 `forward(x, edge_index, aug1, aug2, bs, params)`
- 新增 `embed()` 方法对齐 pygfm baseline 模式
- 三种 loss、EMA 更新、align_reg 逻辑完全保留原版

**注意**：本步中 `NodeNodeContrastiveLoss` / `sample_negative_pairs` 为可选替代方案，当前实现保留了 GIT 原版的 loss 计算逻辑以保证一致性。

### Step 4 — Pretraining Data (Task-Tree Construction)

| pygfm 模块 | 用途 |
|-----------|------|
| _本步未直接使用 pygfm_ | VirtualNodeAugmentor / unified_data 为 GIT 特有算法核心 |

**未使用 pygfm 的原因**：
- `VirtualNodeAugmentor` 是 GIT 论文的核心创新——Task-Tree 构建。它根据不同任务类型（node/edge/graph）向图中注入虚拟任务节点，pygfm 无此概念
- `preprocess` / `postprocess` 处理 GIT 特有的索引-文本嵌入分离存储格式
- `preprocess_data_dict` 实现多数据集全局索引唯一化，为 GIT 特有
- 数据集注册表 (`pretrain_datasets`, `domain2task`, `dataset2domain`) 定义了预训练的多域组合

**与 GIT 原版异同**：
- VirtualNodeAugmentor 逻辑完全一致——保留 GIT 核心算法
- `preprocess` 中分子数据处理从 `MolOFADataset.data` 提取 `pre_edge_index` 和 `node_embs`
- 移除未使用的 `pandas`、`random` 顶层导入，改为惰性导入

### Step 1 — Utils

| pygfm 模块 | 用途 |
|-----------|------|
| `pygfm.public.utils.set_seed` | 全局随机种子 |
| `pygfm.public.utils.early_stopping` | 早停判断（EarlyStopping 类内部引用） |
| `pygfm.public.utils.compute_prototypes` | 计算类别原型嵌入 |

## 源代码来源

每个文件的来源和与 GIT 原版的修改对比。

| 文件 | 来源 | 与原版差异 |
|------|------|-----------|
| `.gitignore` | **新编** | — |
| `README.md` | **新编** | — |
| `__init__.py` | **新编** | — |
| `config/base.yaml` | 复制自 `GIT-main/config/base.yaml` | 无修改 |
| `config/base_zero_shot.yaml` | 复制自 `GIT-main/config/base_zero_shot.yaml` | 无修改 |
| `config/zero_shot.yaml` | 复制自 `GIT-main/config/zero_shot.yaml` | 无修改 |
| `config/in_context.yaml` | 复制自 `GIT-main/config/in_context.yaml` | 无修改 |
| `utils/__init__.py` | **新编** | — |
| `utils/args.py` | 搬运自 `GIT-main/utils/args.py` | 无修改 |
| `utils/eval.py` | 搬运自 `GIT-main/utils/eval.py` | 删除未使用的注释代码块（~100行）；`eval_auc` 函数 NaN 处理调整 |
| `utils/logger.py` | 搬运自 `GIT-main/utils/logger.py` | 无修改 |
| `utils/early_stop.py` | **混合**：搬运 `GIT-main/utils/early_stop.py` + 新增 pygfm 集成 | 新增 `try import pygfm...` 块，声明 `_HAS_PYGFM_EARLY_STOP`（但核心逻辑保留原版） |
| `utils/utils.py` | 搬运自 `GIT-main/utils/utils.py` | 删除 `visualize()`、`sample_proto_instances()`、`sample_proto_instances_for_graph()` 等暂未使用的函数；保留 10 个核心工具函数 |
| `model/__init__.py` | **新编** | — |
| `model/encoder.py` | **混合**：`NonParamPooling` 搬运自 `GIT-main/model/encoder.py`；`GITEncoder` 为改写 | `GITEncoder` 用 pygfm 的 `GraphSAGEEncoderSparse` / `GCNEncoderSparse` / `GATEncoderSparse` 替代原版 `MySAGEConv`；删除 `GATConv` 导入和 `MLP` 类（MLP 移到 decoders.py）；新增 docstring |
| `model/decoders.py` | **混合**：`InnerProductDecoder` 搬运、`MLP` 搬运自 `GIT-main/model/encoder.py` | 两个类从 `encoder.py` 拆分出来独立文件；新增 docstring；`torch.cat` → `.t()` 固定写法 |
| `model/git_pretrain.py` | **改写**：基于 `GIT-main/model/pretrain_model.py` | 父类 `nn.Module` → `GFMPrePromptModelBase`；新增 `gfm_family = "git"`；`forward()` 参数从列表式 `(graph, aug_g1, aug_g2)` → 命名式 `(x, edge_index, aug1, aug2, bs, params)`；新增 `embed()` 方法；新增 docstring |
| `data/__init__.py` | **新编** | — |
| `data/finetune_data.py` | 搬运自 `GIT-main/data/finetune_data.py` | `get_snapshot()` 中 `int()` 改为 `.item()`（适配新版 PyG 类型检查）；删除顶层 `import pandas as pd`，改为 `temporal_graph()` 内惰性导入 |
| `data/ofa_dataset.py` | 搬运自 `GIT-main/data/ofa_dataset.py` | 删除未使用的 `Optional`、`Tuple`、`List` 导入 |
| `data/pretrain_data.py` | 搬运自 `GIT-main/data/pretrain_data.py` | 删除未使用的 `os`、`math`、`pandas`、`random`、`NormalizeFeatures`、`RemoveIsolatedNodes` 导入；`groups.max()` → `groups.max().item()`；新增模块和类的 docstring |
| `task/__init__.py` | **新编** | — |
| `task/node.py` | 搬运自 `GIT-main/task/node.py` | 导入路径调整为本项目；`sft_node` 中 `class_node_text_feat` 索引处理保留原版逻辑 |
| `task/edge.py` | 搬运自 `GIT-main/task/edge.py` | `temporal_datasets` 从 `data.pretrain_data` 导入 |
| `task/link_pred.py` | 搬运自 `GIT-main/task/link_pred.py` | 提取 `predict` 函数；`negative_sampling` 参数换行 |
| `task/graph.py` | 搬运自 `GIT-main/task/graph.py` | `multitask_cross_entropy` 保留；`eval_graph_few_shot` 长行拆分 |
| `pretrain.py` | 搬运改写自 `GIT-main/pretrain.py` | `Encoder` → `GITEncoder`；`PretrainModel` → `GITPrePromptModel`；`seed_everything` → `pygfm.set_seed`；wandb 可选导入；`dropout_adj` → `dropout_edge` |
| `sft.py` | 搬运改写自 `GIT-main/sft.py` | `Encoder` → `GITEncoder`；`seed_everything` → `pygfm.set_seed`；wandb 可选导入；修复原版缩进语法错误 |
| `finetune.py` | 搬运改写自 `GIT-main/finetune.py` | `Encoder` → `GITEncoder`；`TaskModel` → `GITDownPromptNodeModel`/`GraphModel`；`seed_everything` → `pygfm.set_seed`；wandb 可选导入；yaml 惰性导入 |
| `tests/test_step1_utils.py` | **新编** | — |
| `tests/test_step2_model.py` | **新编** | — |
| `tests/test_step3_data.py` | **新编** | — |
| `tests/test_step4_pretrain_data.py` | **新编** | — |
| `tests/test_step5_pretrain_model.py` | **新编** | — |

### 分类统计

| 类别 | 数量 | 说明 |
|------|------|------|
| 复制（无修改） | 5 | 4 个 YAML + `args.py` + `logger.py` |
| 搬运（少量修改） | 5 | `eval.py`、`utils.py`、`finetune_data.py`、`ofa_dataset.py`、`pretrain_data.py` |
| 混合（搬运+改写） | 7 | `early_stop.py`、`encoder.py`、`decoders.py`、`git_pretrain.py`、`pretrain.py`、`sft.py`、`finetune.py` |
| 新编 | 7 | `.gitignore`、`README.md`、5 个 `__init__.py`、5 个测试文件 |
| 待创建 | 0 | — |

## 项目结构

```
main/
├── README.md
├── __init__.py
├── .gitignore
├── pretrain.py                 ├── sft.py                      ├── finetune.py                 ├── utils/
│   ├── __init__.py
│   ├── args.py
│   ├── eval.py
│   ├── logger.py
│   ├── early_stop.py
│   └── utils.py
├── model/
│   ├── __init__.py
│   ├── encoder.py
│   ├── decoders.py
│   ├── git_pretrain.py
│   └── git_downstream.py       ├── data/
│   ├── __init__.py
│   ├── finetune_data.py
│   ├── ofa_dataset.py
│   └── pretrain_data.py
├── task/                        │   ├── __init__.py
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
    ├── test_step1_utils.py       (19 tests)
    ├── test_step2_model.py       (20 tests)
    ├── test_step3_data.py        (13 tests)
    ├── test_step4_pretrain_data.py (18 tests)
    └── test_step5_pretrain_model.py (13 tests)
```
