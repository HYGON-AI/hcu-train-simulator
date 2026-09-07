# Megatron-LM Operator Benchmark

这个目录用于在真实训练环境中测试 Megatron-LM 训练常见算子耗时，补足当前模拟器里 `2 * M * N * K / FLOPS` 这类理论估算和真实 kernel 性能之间的差距。

## 覆盖的常用算子

- Transformer 公共路径：`linear_qkv -> te.pytorch.LayerNormLinear`、`linear_proj -> te.pytorch.Linear`、`TENorm -> te.pytorch.RMSNorm`、`lm_head -> torch.nn.Module`。
- Dense MLP 路径：`linear_fc1 -> te.pytorch.LayerNormLinear`、SwiGLU、`linear_fc2 -> te.pytorch.Linear`。
- Attention 路径：RoPE、attention score softmax/dropout、bias/dropout/residual add、PyTorch SDPA、Flash Attention forward/backward。
- 归一化和融合层：LayerNorm、RMSNorm、Transformer Engine `Linear`、`LayerNormLinear`、`RMSNorm`。
- MoE 路径：`Router(MegatronModule)` 的 router/gate linear、softmax、top-k，以及本地 expert MLP。开启 grouped GEMM 时 FC1/FC2 使用 `te.pytorch.GroupedLinear`；关闭时使用 `te.pytorch.Linear` 循环；共享专家使用 `te.pytorch.Linear`。
- 训练尾部：cross entropy、AdamW optimizer step。

通信类算子也会显著影响 Megatron-LM 训练时间，但它们更适合用已有通信仿真或单独的分布式 benchmark 覆盖：TP all-reduce / all-gather / reduce-scatter、PP send/recv、DP reduce-scatter/all-gather、MoE all-to-all。

## 运行示例

先在没有训练依赖的机器上确认用例形状：

```shell
python -m hcu_train_simulator.benchmarks.operators --preset tiny --dry-run
python -m hcu_train_simulator.benchmarks.operators --preset qwen3_5_9b_vlm --dry-run
```

`qwen3_5_9b_vlm` additionally covers every Qwen3.5-specific estimator row:

- GEMM: gated-attention Q/gate, K, V and output projections; all Gated DeltaNet
  projections; vision patch/QKV/projection/MLP/merger projections.
- Non-GEMM: partial mRoPE, attention output gate, depthwise causal Conv1d,
  GDN gate preprocessing and Q/K L2 norm, the FLA gated-delta-rule kernel,
  gated RMSNorm, vision LayerNorm/RoPE/GELU/residual/position operations, and
  non-causal vision Flash Attention.

The `fla_gated_delta_rule` case is reported as skipped when the target training
environment does not provide the FLA kernel; analytical estimation and profile
fallback remain available.

在完整 PyTorch + Transformer Engine + Flash Attention 训练环境里运行：

```shell
python -m hcu_train_simulator.benchmarks.operators \
  --preset qwen3_30b_a3b_moe \
  --dtype bf16 \
  --tp-size 1 \
  --cp-size 1 \
  --ep-size 8 \
  --etp-size 1 \
  --warmup 20 \
  --iters 100 \
  --output-json benchmark_results/megatron_ops.json \
  --output-csv benchmark_results/megatron_ops.csv
```

只测试某几类算子：

```shell
python -m hcu_train_simulator.benchmarks.operators --ops torch,fa,te --iters 50
python -m hcu_train_simulator.benchmarks.operators --ops te_linear_qkv,flash_attention --iters 100
python -m hcu_train_simulator.benchmarks.operators --ops moe --ep-size 8 --etp-size 1 --moe-grouped-gemm
python -m hcu_train_simulator.benchmarks.operators --ops moe --ep-size 8 --etp-size 1 --no-moe-grouped-gemm
```

并行参数对单卡算子形状的影响：

- `tp-size` 会切分 attention heads、QKV/MLP/LM head 的局部矩阵维度。
- `cp-size` 会切分 sequence/context，脚本将 `seq_length` 视为全局序列长度，并用 `local_seq_length = seq_length / cp_size` 生成单卡局部 token shape。
- `ep-size` 会切分本地 expert 数量，影响 MoE expert MLP loop 的本地 expert 数。
- `etp-size` 会切分 MoE expert FC1/FC2 的中间维度；未指定时默认使用 `tp-size`。
- `dp-size` 和 `pp-size` 通常不改变单层单卡算子 shape，主要影响算子重复次数、pipeline 调度、梯度同步和通信，因此没有放进这个单算子 benchmark。

共享专家可以通过 `--num-shared-experts` 和 `--shared-expert-ffn-hidden-size` 打开。MoE expert 默认覆盖本卡全部 local experts；只做小规模功能烟测时，可以通过 `--max-local-experts` 限制实际 benchmark 数量。

CP attention 默认测试本地 KV chunk。如果想估算 full KV context 下 attention kernel 的形状和耗时，可以加：

```shell
python -m hcu_train_simulator.benchmarks.operators --cp-size 4 --attention-kv-context global --ops torch_sdpa_attention,flash_attention
```

`flash_attention` 和 `te_*` 用例会在对应包不可用或当前设备不支持时标记为 `skipped`，不会影响其它算子的测试。默认不允许 CPU 跑真实 benchmark；如只做功能烟测，可加 `--allow-cpu --device cpu --preset tiny --iters 2`。
