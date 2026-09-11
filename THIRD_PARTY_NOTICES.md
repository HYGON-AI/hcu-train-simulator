# Third-Party Notices

本文件记录 `hcu-train-simulator` 在构建、基础运行和可选算子实测过程中使用或引用的第三方组件。

本仓库不内嵌第三方源码副本（无 `vendor/`、`third_party/` 或 `external/` 目录）；Python 依赖由安装工具根据 `pyproject.toml` 获取。可选算子实测组件由用户的训练环境提供，不随本仓库分发。

本项目许可证：`Apache-2.0`（见 [LICENSE](LICENSE)）。  
Copyright (c) 2026 Hygon Information Technology Co., Ltd.

## Direct Python runtime dependencies

以下为 `pyproject.toml` 中声明的直接运行时依赖。

| 项目 / 包 | 上游仓库 | 版本约束 | Copyright（上游声明） | 许可证 | 本地路径 | HYGON 修改 |
| --- | --- | --- | --- | --- | --- | --- |
| PyYAML | https://github.com/yaml/pyyaml | `PyYAML==6.0.3` | Copyright (c) 2017-2021 Ingy döt Net；Copyright (c) 2006-2016 Kirill Simonov | MIT | Python 环境的 `site-packages`；源码树中不内嵌 | 无 |
| tabulate | https://github.com/astanin/python-tabulate | `tabulate==0.10.0` | Copyright (c) 2011-2020 Sergey Astanin and contributors | MIT | Python 环境的 `site-packages`；源码树中不内嵌 | 无 |

## Build dependency

| 项目 / 包 | 上游仓库 | 版本约束 | 许可证 | 用途 | HYGON 修改 |
| --- | --- | --- | --- | --- | --- |
| setuptools | https://github.com/pypa/setuptools | `setuptools>=61` | MIT；发行包可能携带具有独立许可证的 vendored 组件，以实际安装版本随附的许可证文件为准 | PEP 517 构建后端 `setuptools.build_meta` | 无 |

## Optional operator-benchmark environment

下列组件不属于基础安装依赖。仅当用户启用真实算子 Profile/Benchmark 时，相关代码才会按需导入或探测它们；未安装时，模拟器可使用内置 Profile 或理论估算。

| 项目 / 组件 | 公开上游 | 仓库中的版本状态 | 公开上游许可证 | 用途 | HYGON 修改 / 分发边界 |
| --- | --- | --- | --- | --- | --- |
| PyTorch | https://github.com/pytorch/pytorch | 未在 `pyproject.toml` 固定；内置 Profile 记录环境版本 `2.7.1` | BSD-3-Clause 为主，发行包包含其他第三方许可证 | HCU 训练环境探测及算子实测 | 本仓库不分发 PyTorch；实际 HCU 构建的补丁和许可证以环境提供方材料为准 |
| NVIDIA Transformer Engine | https://github.com/NVIDIA/TransformerEngine | 未作为安装依赖固定；内置 Profile 记录环境版本 `2.10.0+das.opt1.dtk2604.torch271` | Apache-2.0 | TE Linear、Norm、GroupedLinear、RoPE 等算子实测 | 本仓库不分发该组件；Profile 中的 HCU 构建仅作为实测环境记录 |
| FlashAttention | https://github.com/Dao-AILab/flash-attention | 未作为安装依赖固定；内置 Profile 记录环境版本 `2.6.1+das.opt1.dtk2604.torch271` | BSD-3-Clause | Flash Attention 算子实测 | 本仓库不分发该组件；Profile 中的 HCU 构建仅作为实测环境记录 |
| Megatron-LM / Megatron Core | https://github.com/NVIDIA/Megatron-LM | 未固定；仅在部分 RoPE benchmark 路径中按需导入 | BSD-3-Clause 为主；仓库中部分文件保留其他许可证声明 | Megatron 风格算子和 RoPE 实测接口 | 本仓库不分发 Megatron-LM 源码；模拟器主体不依赖其运行 |

## Interface and data-source boundary

- 模拟器读取用户提供的 Hugging Face 风格 `config.json`，但不依赖或内嵌 Hugging Face Transformers 源码。
- `src/hcu_train_simulator/modeling/` 使用 Megatron-Core 风格的字段和轻量语义标记；这些类型由本仓库自行实现，不是 Megatron-LM 源码副本。
- `src/hcu_train_simulator/profiles/builtin/` 保存 HYGON 环境中的算子实测数据和环境指纹，不包含 PyTorch、Transformer Engine、FlashAttention、Triton 或 DTK 的源码或二进制。
- `deepep` 是模拟器内置的通信阶段启发式模型名称；本仓库不导入或内嵌 DeepEP 项目源码。

## Compliance boundary

- 本仓库源码适用 `Apache-2.0` 与 HYGON Copyright；第三方组件继续适用其各自许可证。
- 不在本仓库源码中机械添加第三方 Copyright 或 SPDX；第三方归属以本清单及组件发行包随附的 LICENSE/NOTICE 为准。
- 构建、安装或打包工具应保留实际安装依赖随附的许可证和通知文件。
- 若 `pyproject.toml` 的依赖、版本约束、可选算子后端或内置 Profile 环境发生变化，应同步更新本清单。
