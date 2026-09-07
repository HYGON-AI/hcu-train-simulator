# hcu-train-simulator

## 1. 介绍

本仓库提供一个面向大模型训练的性能估算工具，可根据 HuggingFace 格式的模型 `config.json`、Megatron-LM 风格的并行策略以及硬件参数，估算一次训练迭代中的显存占用、计算时间、通信时间和整体吞吐表现。工具以 `hcu-train-sim` 命令行形式对外提供能力，支持单项分析和端到端模拟，并可生成 HTML 训练模拟报告。

主要功能包括：

- **显存占用模拟**：按 Pipeline Parallel stage 输出参数、梯度、优化器状态和激活值等显存占用，支持 TP、CP、PP、VPP、EP、ETP 等并行配置对显存的影响估算。
- **计算时间模拟**：基于模型结构、batch、序列长度、并行切分、FP16 峰值算力和计算效率，估算 dense / MoE Transformer 中 QKV、Attention Projection、MLP / MoE MLP、Router、LM Head、Flash Attention，以及 RMSNorm、RoPE、SwiGLU、residual/dropout/add、loss、optimizer step 等非 GEMM 模块的前向和反向计算时间。
- **通信时间模拟**：按 TP、CP、DP、PP 通信域统计通信耗时；MoE 模型额外统计 EP、ETP 通信耗时。通信模型会结合节点内/节点间带宽、通信类型、并行拓扑以及通信掩盖比例进行估算。
- **端到端训练模拟**：汇总显存、计算、通信结果，输出模型规模、计算时间、通信时间、单迭代总时间、TGS（tokens/p/s）、MFU 和通信占比等关键指标。
- **最优性能网格搜索**：根据 `search.candidates` 中配置的并行策略和 micro batch 候选组合进行网格搜索，默认按最高 TGS 排序，过滤超过显存上限的组合，并在终端输出 Top 10 结果。
- **报告生成**：`run` 模式会在当前目录下生成 `training_report/training_report_<timestamp>.html`，报告中包含关键指标卡片、各 PP stage 显存、计算明细以及通信域耗时分布。

估算原理概览：

- 显存估算将模型参数量拆分到不同 PP stage，并结合 DP/CP/EP/ETP、分布式优化器、激活值和 micro batch 数量计算静态显存与动态显存。
- 计算时间估算以主要矩阵乘算子为核心，按 `2 * M * N * K / (fp16_tflops * gemm_efficiency)` 近似 GEMM 耗时；非 GEMM vector/elementwise 操作按元素数、每元素操作系数和 `non_gemm_efficiency` / `optimizer_efficiency` 估算，并叠加到前向、反向或 optimizer step 耗时中。
- 通信时间估算按不同通信域构造 AllReduce、AllGather、ReduceScatter、AllToAll、P2P 等通信量，并结合带宽表和节点内/节点间带宽得到耗时。

注意：当前计算时间模拟覆盖 GEMM、Flash Attention 以及一组常见非 GEMM 开销，包括 RMSNorm / QK Norm、RoPE、Attention Softmax、SwiGLU、Residual/Dropout/Add、Cross Entropy 和 AdamW optimizer step。数据搬移、Kernel Launch、框架调度、未列出的 fused kernel 细节仍未精确建模，因此端到端时间可用于方案对比和瓶颈分析，但不等价于真实训练作业的精确运行时间。

### 1.1 代码结构

```text
src/hcu_train_simulator/
├── benchmarks/      # 算子实测与环境探测
├── commands/        # CLI 子命令
├── communication/   # 拓扑、通信组、通信流、算法与带宽表
├── config/          # 配置模型、加载与校验
├── templates/       # 可复制的配置模板
├── estimators/      # 显存、计算和通信估算入口
├── modeling/        # 参数、激活、计算和模型统计
├── reporting/       # HTML 渲染与报告写入
├── parallelism.py   # 并行切分公共逻辑
├── context.py       # 进程内模拟配置上下文
└── simulator.py     # 端到端训练模拟编排
```

仓库根目录的 `train_sim.py` 保留为开发验证入口，安装后的正式命令为 `hcu-train-sim`。


## 2. 模型及配置支持

支持的配置参数可参考 `src/hcu_train_simulator/templates/config.yaml` 配置文件

### 2.1 模型支持

|    参数名     | 默认值 |   参数说明   |
|:----------:|:---:|:--------:|
| model_path |  /  | 模型配置文件路径 |

`model_path` reads a HuggingFace-style `config.json`. In addition to dense and
standard MoE fields, the simulator now recognizes these architecture fields:
`padded_vocab_size`, `make_vocab_size_divisible_by`, `vocab_size_multiple`,
`num_experts`/`n_routed_experts`, `num_experts_per_tok`/`topk`,
`num_shared_experts`/`n_shared_experts`, `shared_expert_intermediate_size`,
MLA fields `q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`,
`qk_rope_head_dim`, `v_head_dim`, and MTP fields `mtp_num_layers`,
`num_mtp_layers`, `num_nextn_predict_layers`.GLM-5 DSA configs additionally
use `index_n_heads`, `index_head_dim`, `index_topk`, and optional per-layer
`indexer_types` (`full` or `shared`).

GLM-5 (`model_type: glm_moe_dsa`) is modeled as MLA + DeepSeek Sparse
Attention + dense/MoE FFN + MTP. The compute breakdown includes the replicated
lightning-indexer Q/K/head-weight projections, index score matmul, key
LayerNorm, indexer RoPE, score reduction, token top-k, and the `O(L * topk)`
sparse MLA kernel. Parameter and activation estimates include indexer weights,
saved top-k indices, and the fused index-score workspace. Following the public
GLM-5/DeepSeek implementation, indexer weights and indexer compute are treated
as replicated across TP ranks; no extra TP collective is charged for them.
`indexer_types` can describe IndexShare-style layers that reuse a previous full
indexer's top-k selection.

Qwen3.5 VLM checkpoints are accepted directly in their nested HuggingFace form
(`text_config` + `vision_config`). The resolved Megatron-style spec models the
3:1 Gated DeltaNet / gated-softmax-attention layer pattern, partial mRoPE,
attention output gates, MTP, dense or MoE FFNs (including the shared-expert
gate), the vision patch encoder, all vision Transformer blocks, and the patch
merger. `parallel_config.vision_seq_length` controls visual patch tokens per
image; when it is `null`, `vision_config.num_position_embeddings` is used.
`parallel_config.vision_num_images` defaults to `1`.

Qwen3-VL dense and MoE checkpoints are also accepted directly in nested
HuggingFace form. The adapter recognizes `qwen3_vl` / `qwen3_vl_moe`, QK Norm,
mRoPE metadata, `decoder_sparse_step`, `mlp_only_layers`, the ViT patch encoder,
all visual Transformer blocks, the primary patch merger, and every DeepStack
merger listed by `vision_config.deepstack_visual_indexes`. The DeepStack
parameters, activations, merger compute, and language-model feature additions
are included in the estimates. A runnable Qwen3-VL-30B-A3B example is provided
at `examples/qwen3_vl_30b_a3b/simulation.yaml`.

支持 HuggingFace 格式的 `config.json` 文件，`model_path` 可以指向 `config.json` 文件，也可以指向包含 `config.json` 的模型目录。如果是自定义模型，可基于现有开源模型配置修改后传入。dense 模型和包含专家参数的 MoE 模型均可进行模拟。

### 2.2 训练参数配置


|                参数名                |  默认值  |            参数说明            |
|:---------------------------------:|:-----:|:--------------------------:|
|              tp_size              |   1   |            张量并行            |
|              cp_size              |   1   |           上下文并行            |
|              ep_size              |   1   |            专家并行            |
|             etp_size              | null  |          专家域张量并行           |
|              pp_size              |   1   |            流水并行            |
|            pp_schedule            | 1f1b  | 流水调度方式，支持 `1f1b`、`interleaved`、`interleaved_1f1b` |
|      num_layers_per_vp_stage      | null  |   每个虚拟流水线的transformer层数    |
|     use_distributed_optimizer     | True  |          使用分布式优化器          |
| decoder_first_pipeline_num_layers | null  | 第一个pp stage的transformer层数  |
| decoder_last_pipeline_num_layers  | null  | 最后一个pp stage的transformer层数 |
|          full_recompute           | False |         是否开启全量重计算          |
|         micro_batch_size          |   1   |       每个模型实例的局部微批次大小       |
|         global_batch_size         |  256  |          全局训练批次大小          |
|            seq_length             | 4096  |         待处理的最大序列长度         |
|             num_gpus              |   8   |         训练使用的GPU数量         |
|          overlap_mode             | auto  | 通信掩盖模式，支持 `auto`、`manual`、`none` |
|       ep_overlap_enabled          | false | `auto` 模式下是否允许 EP dispatch/combine 被专家计算掩盖 |
|    ep_communication_backend       | alltoall | EP 通信后端，支持 `alltoall` 和内置启发式 `deepep` 四阶段模型 |
|         tp_overlap_ratio          |   0   | `manual` 模式下的 TP 域通信掩盖比例 |
|         cp_overlap_ratio          |   0   | `manual` 模式下的 CP 域通信掩盖比例 |
|         dp_overlap_ratio          |   0   | `manual` 模式下的 DP 域通信掩盖比例 |
|         ep_overlap_ratio          |   0   | `manual` 模式下的 EP 域通信掩盖比例 |
|         pp_overlap_ratio          |   0   | `manual` 模式下的 PP 域通信掩盖比例 |

根据需要模拟的训练配置来填写训练参数。`decoder_first_pipeline_num_layers` 和 `decoder_last_pipeline_num_layers` 用于描述首尾 PP stage 的自定义层数，会影响 PP stage 显存、通信和调度气泡等估算。

### 2.3 硬件参数配置

|            参数名             | 默认值 |    参数说明    |
|:--------------------------:|:---:|:----------:|
|       fp16_tflops       | 480 | FP16峰值算力（TFLOPS） |
|       fp8_tflops        | 960 | FP8峰值算力（TFLOPS） |
|      gpus_per_node      |  8  | 单节点GPU卡数 |
|         hbm_gib         | 64  | 单卡HBM容量（GiB） |
|      intra_bw_gbps      | 448 | 节点内互联带宽（GB/s） |
|      inter_bw_gbps      | 128 | 节点间互联带宽（GB/s） |
|    gemm_efficiency      | 0.4 | GEMM计算效率 |
|    non_gemm_efficiency  | 0.08 | 非 GEMM vector/elementwise 计算效率 |
|    optimizer_efficiency | 0.04 | Optimizer step 计算效率 |
| p2p_intra_efficiency    | 0.8 | 节点内P2P通信效率 |
| collective_intra_efficiency | 0.7 | 节点内集合通信效率 |
| collective_inter_efficiency | 0.8 | 节点间集合通信效率 |

### 2.4 搜索参数配置

`search` 模块用于配置最优 TGS 网格搜索。当前目标函数固定为最高 TGS，不开放配置项供用户切换；搜索结果只在终端打印，不写入 HTML 报告。

| 参数名 | 默认值 | 参数说明 |
|:---:|:---:|:---|
| enabled | true | 是否允许执行 `hcu-train-sim search config.yaml` |
| top_k | 10 | 终端输出的结果数量，当前最多保留 Top 10 |
| memory_margin_gib | 5 | 显存安全余量，候选组合的峰值显存不能超过 `hbm_gib - memory_margin_gib` |
| candidates | / | 网格搜索候选空间，目前支持 `tp_size`、`cp_size`、`pp_size`、`ep_size`、`etp_size`、`micro_batch_size` 等并行和批次参数 |

示例：

```yaml
search:
  enabled: true
  top_k: 10
  memory_margin_gib: 5
  candidates:
    tp_size: [1, 2, 4, 8]
    cp_size: [1]
    pp_size: [1, 2, 4, 8]
    ep_size: [1, 2, 4, 8]
    etp_size: [1, 2, 4]
    micro_batch_size: [1, 2, 4]
```


## 3. 使用方法

### 3.1 安装

```shell
# clone 仓库
git clone http://42.228.13.241:10068/yuhui1/hcu-train-simulator.git

# build whl 包
cd hcu-train-simulator
python -m build

# 安装whl包
cd dist
pip install hcu_train_simulator-*.whl
```

### 3.2 使用

在使用`hcu-train-sim`工具进行大模型训练性能模拟时，应先执行：

```shell
hcu-train-sim init
```

命令执行成功后，会在当前目录下生成一个`config.yaml`配置文件，按照实际需求，进行参数配置，参数定义可参考上述内容。

#### 3.2.1 系统模拟

```shell
hcu-train-sim run config.yaml
```

开发验证时可临时关闭 HTML 报告生成：

```shell
hcu-train-sim run config.yaml --no-report
python train_sim.py config.yaml --no-report
```

执行 `hcu-train-sim run config.yaml` 后，会全量模拟显存占用、通信时间以及计算时间，并在当前目录创建 `training_report` 目录（如有则不创建），在 `training_report` 目录下生成一份 HTML 格式的模拟报告以供查看详细信息。带 `--no-report` 参数时只输出终端指标，不生成 HTML 报告。

#### 3.2.2 显存占用模拟

```shell
hcu-train-sim analyze memory config.yaml
```

执行上述命令后，仅模拟显存占用情况，并在终端打印每个pp stage的显存占用量，包含静态显存和动态显存明细。

#### 3.2.3 通信时间模拟

```shell
hcu-train-sim analyze comm config.yaml
```

执行上述命令后，仅模拟通信时间情况，并在终端打印每个通信域的具体耗时。

#### 3.2.4 计算时间模拟

```shell
hcu-train-sim analyze compute config.yaml
```

执行上述命令后，仅模拟计算时间情况，并在终端打印每个transformer模块的具体耗时。

#### 3.2.5 最优性能网格搜索

```shell
hcu-train-sim search config.yaml
```

执行上述命令后，会读取 `config.yaml` 中的 `search.candidates` 生成候选组合，对每个有效组合执行显存、计算和通信估算，并按 TGS 从高到低输出最多 10 条结果。超过 `hbm_gib - memory_margin_gib` 的组合会被过滤；无效并行组合（例如 GPU 数量不能整除并行维度、TP 不能整除 KV Head、PP 不能整除层数等）会被跳过。

#### 3.2.6 算子性能 Profile

工具包含只读的内置算子 Profile，资源位于 Python 包的
`hcu_train_simulator/profiles/builtin/` 下。`hcu-train-sim init` 只生成
`config.yaml`，不会复制或创建 Profile 文件。

Profile 可通过 `profile_config.mode` 选择以下模式：

- `auto`：检测当前 Torch/DCU 训练环境，按算子 shape 依次查询用户 Profile、内置 Profile，并只对缺失 shape 进行实测。
- `builtin`：没有训练环境时，根据 `builtin_selector` 显式选择内置 Profile。当前 selector 支持 `bw1000`/`bw1100`、Torch `271`/`290` 和 DTK `2604`。
- `theoretical`：不使用任何实测 Profile，全部使用理论计算结果。

用户 Profile 默认写入 `./profiles/user/`。相对路径以 `config.yaml` 所在目录为基准；目录只在第一次需要记录实测数据时创建。文件名包含硬件、Torch、DTK 以及完整环境指纹，例如：

```text
bw1000_torch271_dtk2604_a84c31d2.yaml
```

环境信息完整保存加速器型号、Torch、Transformer Engine、Flash Attention、Triton、DTK、驱动和 dtype。DTK 版本从 `${ROCM_PATH}/.dtk_version` 第一行读取；文件名使用 `2604` 形式的版本键，但 Profile 内保留 `DTK-26.04-rc4` 等完整原始版本。软件版本只保存在 `environment.software` 中。

每个算子 shape 独立维护可信状态。数据查询优先级为：

```text
user validated > builtin validated > clean live benchmark > theoretical
```

GEMM 和 Flash Attention 的 validated shape 只保存 `status`、`mean_ms`、`tflops` 和 `updated_at`；non-GEMM shape 不保存 `tflops` 字段。candidate 记录额外保存简短的 `candidate_reason`。不保存采样次数、最大/最小耗时、计时离散度或其他冗长诊断信息。

自动实测前后会检查加速器占用情况。测试前发现其他进程时跳过实测；测试后目标卡的 HCU 利用率非零或占用状态无法确认时，结果只会保存为 `candidate`。单算子计时离散度不影响结果状态。终端会列出本次没有 validated 的算子、shape 和具体原因。Candidate 不参与本次或后续性能估算，只用于下次环境空闲时提示重新实测，并且不会覆盖已有的 validated 数据。

终端和 HTML 报告会区分 `user profile hit`、`builtin profile hit`、`live benchmark` 和 `theoretical`，同时给出忽略的 candidate 数量。内置 Profile 只做精确 selector 匹配，不会自动选择“最接近”的软件版本。

## 4. 扩展新模型

新增模型优先通过 `src/hcu_train_simulator/models/<model>/` 下的适配器和能力注册完成。配置归一化、ModuleSpec 组合、模型特有成本公式以及 GEMM/non-GEMM 实测映射均可保留在模型目录中。具体接口和验收清单见 [模型插件开发指南](docs/model_plugins.md)。


