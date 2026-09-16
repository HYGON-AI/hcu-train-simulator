---
name: model-training-simulator
description: Estimate large language model training performance with the hcu-train-sim wheel/CLI. Use when Codex needs to configure and run hcu-train-simulator for GPU/HCU memory usage, compute time, communication time, iteration time, TGS, MFU, parallel strategy comparison, grid search, or HTML training simulation reports for dense or MoE Transformer models.
---

# Model Training Estimator

Use this skill to call the `hcu-train-sim` package produced by the local `hcu-train-simulator` repository. The package estimates LLM training memory, major compute time, communication time, iteration time, TGS, MFU, supports TGS-oriented grid search, and generates an HTML report.

## Quick Contract

Use the installed CLI, not the old source scripts. The normal flow is:

```shell
pip install /path/to/hcu_train_simulator-*.whl
hcu-train-sim init
# edit config.yaml
hcu-train-sim run config.yaml
```

Run CLI commands from the working directory where the generated `config.yaml` and `training_report/` output should live. `hcu-train-sim init` copies the packaged template to `./config.yaml` and overwrites an existing file with the same name, so back up user configs before running it. The config points to `model_path`, which can be either a model directory containing `config.json` or a direct path to `config.json`.

Preserve user changes. Before editing an existing config file, read it first and decide whether to make a timestamped backup or restore it after a one-off run. If the user asks for a scenario comparison, run scenarios one by one by updating a copied YAML file and saving each command output/report path clearly.

## Minimal Inputs

Before running, make sure these are known or inferable:

- Model config: a HuggingFace-style `config.json`; MoE configs should include expert fields.
- Training/parallel strategy: GPU count, batch/sequence length, and TP/CP/PP/EP/ETP settings.
- Hardware assumptions: FLOPS, HBM capacity, intra/inter-node bandwidth, and efficiency coefficients.

Ask a concise question only if the model config path or essential model dimensions are unavailable. If the user supplies dimensions instead of a `config.json`, create a minimal scenario-specific `config.json` in the workspace and point `model_path` to it.

## Configuration Rules

Generate the template first if needed:

```shell
hcu-train-sim init
```

Edit only the relevant fields in the target YAML file, usually `config.yaml`. The important shape is:

```yaml
model_path: /path/to/model/or/config.json

parallel_config:
  tp_size: 1
  cp_size: 1
  ep_size: 1
  etp_size: null
  pp_size: 1
  pp_schedule: 1f1b
  num_layers_per_vp_stage: null
  use_distributed_optimizer: true
  decoder_first_pipeline_num_layers: null
  decoder_last_pipeline_num_layers: null
  full_recompute: false
  micro_batch_size: 1
  global_batch_size: 256
  seq_length: 4096
  num_gpus: 8
  overlap_mode: auto
  tp_overlap_ratio: 0
  cp_overlap_ratio: 0
  dp_overlap_ratio: 0
  ep_overlap_ratio: 0
  pp_overlap_ratio: 0

hardware_config:
  use_bandwidth_table: true
  fp16_tflops: 480
  fp8_tflops: 960
  gpus_per_node: 8
  hbm_gib: 64
  intra_bw_gbps: 448
  inter_bw_gbps: 128
  gemm_efficiency: 0.4
  p2p_intra_efficiency: 0.8
  collective_intra_efficiency: 0.7
  collective_inter_efficiency: 0.8

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

Respect these code-derived behaviors:

- `dp_size = num_gpus // (tp_size * cp_size * pp_size)`.
- Dense models default `etp_size` to `1` when it is `null`.
- MoE models default `etp_size` to `tp_size` when it is `null`.
- `num_gpus` must be divisible by `tp_size * cp_size * pp_size`.
- `num_key_value_heads` must be divisible by `tp_size`.
- If custom first/last pipeline layer counts are not set, `num_hidden_layers` must be divisible by `pp_size`.
- VPP requires `pp_size > 1` and a valid integer virtual pipeline size.
- `pp_schedule` supports `1f1b`, `interleaved`, and `interleaved_1f1b`; interleaved schedules require `num_layers_per_vp_stage`.
- `overlap_mode` supports `auto`, `manual`, and `none`; manual mode uses the per-domain overlap ratios.
- Communication simulation uses `GPUType.BW1000` and bandwidth tables under `communication/bandwidth`.
- `search` is a CLI-level grid-search block. Its objective is fixed to highest TGS, `top_k` is capped at 10, and `memory_margin_gib` filters candidates whose peak `total_gib` exceeds `hbm_gib - memory_margin_gib`.

## Run Modes

Use the narrowest CLI command that answers the user:

```shell
hcu-train-sim analyze memory config.yaml
```

Returns per-PP-rank memory table with parameter bytes, activation bytes, weight/grad/optimizer GB, activation GB, and total GB.

```shell
hcu-train-sim analyze compute config.yaml
```

Returns per-model-part GEMM, attention, and modeled non-GEMM compute timing with forward and backward milliseconds.

The compute estimator covers dominant GEMM-style and Flash Attention costs plus common non-GEMM work such as RMSNorm/QK Norm, RoPE, attention softmax, SwiGLU, residual/dropout/add, cross entropy, and AdamW optimizer step. Data movement, kernel launch overhead, framework scheduling, and unlisted fused-kernel details are still approximate or not explicitly modeled. Treat compute time and end-to-end iteration time as planning estimates for comparing configurations, not exact training job latency.

```shell
hcu-train-sim analyze comm config.yaml
```

Returns communication timing by TP/CP/DP/PP and, for MoE, EP/ETP. The raw values are seconds in the communication module.

```shell
hcu-train-sim run config.yaml
```

Runs memory, compute, and communication estimators together, validates arguments, logs the final metrics, and writes an HTML report to `training_report/training_report_<timestamp>.html`.

```shell
hcu-train-sim search config.yaml
```

Runs a grid search over `search.candidates`, evaluates valid candidates with the same memory/compute/communication estimators, filters by the configured memory margin, and prints terminal-only Top-K results ranked by TGS. The current search command does not generate or update an HTML report.

Prefer `hcu-train-sim run config.yaml` for end-to-end user requests such as "estimate training performance", "generate report", or "what is the MFU/TGS/iteration time". Prefer `hcu-train-sim search config.yaml` when the user asks for best performance, best TGS, or automatic parallel-strategy search. Use the `analyze` subcommands only when the user asks for a single memory, compute, or communication estimate.

## Output To Report

When summarizing a run, extract and report the key metrics:

- Model size in B parameters.
- Peak or per-stage memory, especially `total_gib` versus `hardware_config.hbm_gib`.
- Compute time in ms.
- Communication time in ms.
- Iteration total time in ms.
- TGS in `tokens/p/s`.
- MFU, usually shown as a ratio in terminal logs and percent in the HTML report.
- Communication breakdown and percentages by TP/CP/DP/PP/EP/ETP.
- Path to the generated HTML report when `hcu-train-sim run` is used.

When reporting compute or iteration time, mention the compute-scope limitation if the user is comparing against real training logs or asking for absolute runtime accuracy.

`reporting/html.py` defines the HTML layout. The `hcu-train-sim run` CLI path calls `generate_training_report(...)` with memory stages, compute details, communication domains, and parallel strategy.

## Optimization Advice

After a full training simulation, always provide concise optimization suggestions tied to the measured bottleneck:

- If peak `total_gib` is close to or above `hbm_gib`, suggest lowering `micro_batch_size` or `seq_length`, increasing `tp_size`/`pp_size`/`cp_size` where valid, enabling or verifying recompute support, or using distributed optimizer.
- If communication time is high, identify the dominant domain from TP/CP/DP/PP/EP/ETP and suggest parallelism changes that reduce that domain's group size or cross-node traffic. Mention overlap ratios only as an exposure model, not a guaranteed runtime gain.
- If compute time dominates and MFU is low, suggest checking `gemm_efficiency`, `non_gemm_efficiency`, and `optimizer_efficiency`, using a faster hardware profile, improving GEMM/non-GEMM efficiency, increasing batch utilization, or comparing TP/PP layouts that keep operators larger.
- If PP communication or imbalance is high, suggest checking layer divisibility, custom first/last PP layer counts, VPP, and per-stage memory skew.
- If DP communication dominates, inspect the per-rank parameter estimate and whether distributed optimizer selects ReduceScatter plus AllGather or standard AllReduce.
- For MoE runs, separately inspect EP/ETP communication and expert memory; suggest tuning `ep_size`, `etp_size`, and `num_experts_per_tok` only when compatible with the model and GPU count.

Frame suggestions as next experiments, not guaranteed fixes. Include the reason each suggestion follows from the simulated result.

## Scenario Comparison Workflow

For multiple candidate configurations:

1. Record the base YAML file, usually `config.yaml`.
2. If the user wants automatic search over parallelism and micro batch candidates, put the candidate lists under `search.candidates` and run `hcu-train-sim search config.yaml`.
3. If the user wants hand-picked scenario comparison, update only the relevant YAML fields in scenario-specific copies such as `config_tp4_pp2.yaml`.
4. Run `hcu-train-sim run <scenario-config>.yaml` for detailed scenario metrics and reports.
5. Capture terminal metrics and generated report path, or for search capture the terminal Top-K table.
6. Compare memory feasibility, iteration time, TGS, MFU, and dominant communication domain.
7. Restore the original YAML unless the user asked to keep the final scenario.

Keep a compact comparison table in the final response. Rank configurations by the user's objective, such as fastest iteration time, fits-in-memory, highest MFU, or lowest communication cost.

## Failure Handling

If a run fails:

- Missing `config.json`: verify `model_path`; it may be either the config file or the containing model directory.
- `num_key_value_heads must be divisible by tp_size`: choose a TP size that divides KV heads.
- `num_hidden_layers must be divisible by pp_size`: choose a PP size that divides layers, or set valid `decoder_first_pipeline_num_layers` / `decoder_last_pipeline_num_layers`.
- `num_gpus must be divisible...`: adjust `num_gpus`, `tp_size`, `cp_size`, or `pp_size`.
- VPP validation errors: use `pp_size > 1` and make `num_hidden_layers / pp_size / num_layers_per_vp_stage` integral.
- Search returns no valid results: loosen `search.candidates`, lower `memory_margin_gib`, increase HBM capacity assumptions, or remove candidate values that violate TP/PP/GPU divisibility constraints.
- Unknown search candidate key: only use fields that exist in `parallel_config`; current search does not accept hardware or model-architecture candidates.
- CLI not found: install the wheel into the active Python environment with `pip install /path/to/hcu_train_simulator-*.whl`, then retry `hcu-train-sim --help`.
- Missing packaged template: reinstall the wheel built after the package-data change; `hcu-train-sim init` expects `hcu_train_simulator.templates/config.yaml` inside the installed package.
- Bandwidth table errors: check packaged `communication/bandwidth` files and the `GPUType.BW1000` selection in `estimators/communication.py`.
- Dependency errors: install or activate the wheel environment; runtime dependencies include `pyyaml` and `tabulate`.

Do not silently change model architecture values to make validation pass. Explain the incompatible field and propose the smallest parallelism/config change.

## Implementation Notes

- Use structured YAML editing when possible.
- Use `hcu-train-sim` for installed operation. The retained `python train_sim.py` entry is intended for source-tree development validation.
- Do not initialize multiple configurations in one Python process without calling `context.reset_config()`; separate CLI invocations avoid this issue.
- The search command evaluates many configurations in one process and must reset the process-local context between candidates.
- Search results are intentionally terminal-only for now; do not promise HTML report integration for `hcu-train-sim search`.
- DP communication volume is derived from the maximum per-rank parameter count and the configured optimizer mode.
- `hardware_config.hbm_gib` is an input capacity reference, but feasibility is decided by comparing estimator output `total_gib` with that capacity; the code does not automatically fail when memory exceeds HBM.
- `full_recompute` is present in config but activation formulas should be checked before promising recompute savings.
