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

## 项目结构

```
main/
├── README.md
├── __init__.py
├── pretrain.py
├── sft.py
├── finetune.py
├── utils/
│   ├── args.py          # 三入口 argparse
│   ├── eval.py          # ACC / AUC 评估
│   ├── logger.py        # 多轮训练日志
│   └── early_stop.py    # 早停（封装 pygfm.early_stopping）
├── model/
│   ├── encoder.py       # GITEncoder (pygfm backbone + 自定义 pooling)
│   ├── decoders.py      # InnerProductDecoder, MLP
│   ├── git_pretrain.py
│   └── git_downstream.py
├── data/
│   ├── pretrain_data.py
│   ├── finetune_data.py
│   └── ofa_dataset.py
├── task/
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
    ├── test_step1_utils.py          # Step 1: 19 tests
    ├── test_step2_model.py          # Step 2: 20 tests
    ├── test_step3_data.py           # Step 3: 13 tests
    └── test_step4_pretrain_data.py  # Step 4: 18 tests
```
