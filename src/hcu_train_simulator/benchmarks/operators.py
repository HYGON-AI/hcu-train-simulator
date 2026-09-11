# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark common Megatron-LM training operators on a real PyTorch runtime.

The script is intentionally standalone: --list-ops and --dry-run do not import
torch, while the actual benchmark path imports torch/Transformer Engine/Flash
Attention lazily on the target training node.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Any

from hcu_train_simulator.logging import get_logger, log_table
from hcu_train_simulator.modeling.spec import (
    spec_has_qk_norm,
    spec_uses_dsa,
    spec_uses_gated_delta_net,
    spec_uses_mla,
    spec_uses_mtp,
    spec_uses_vision,
)
from hcu_train_simulator.modeling.architecture import layer_uses_moe
from hcu_train_simulator.benchmarks.registry import (
    benchmark_presets,
    get_benchmark_case,
    registered_plan_builders,
)
from hcu_train_simulator.benchmarks.environment import read_dtk_version


logger = get_logger(__name__)


@dataclass(frozen=True)
class OperatorPlan:
    name: str
    group: str
    description: str
    shape: str
    requires: str = "torch"


@dataclass
class BenchmarkResult:
    name: str
    group: str
    status: str
    mean_ms: float | None
    p50_ms: float | None
    min_ms: float | None
    max_ms: float | None
    tflops: float | None
    peak_memory_mb: float | None
    shape: str
    message: str = ""


PRESETS = {
    "tiny": {
        "hidden_size": 128,
        "ffn_hidden_size": 512,
        "moe_ffn_hidden_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 32,
        "seq_length": 16,
        "micro_batch_size": 1,
        "vocab_size": 1024,
        "num_experts": 4,
        "topk": 2,
        "num_shared_experts": 0,
        "shared_expert_ffn_hidden_size": 256,
        "has_qk_norm": False,
    },
    "qwen2_7b": {
        "hidden_size": 3584,
        "ffn_hidden_size": 18944,
        "moe_ffn_hidden_size": 18944,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "seq_length": 4096,
        "micro_batch_size": 1,
        "vocab_size": 152064,
        "num_experts": 1,
        "topk": 1,
        "num_shared_experts": 0,
        "shared_expert_ffn_hidden_size": 18944,
        "has_qk_norm": False,
    },
    "qwen3_30b_a3b_moe": {
        "hidden_size": 2048,
        "ffn_hidden_size": 6144,
        "moe_ffn_hidden_size": 768,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "seq_length": 4096,
        "micro_batch_size": 1,
        "vocab_size": 151936,
        "num_experts": 128,
        "topk": 8,
        "num_shared_experts": 0,
        "shared_expert_ffn_hidden_size": 768,
        "has_qk_norm": True,
    },
}
PRESETS.update(benchmark_presets())


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Megatron-LM style training operators with PyTorch, Transformer Engine, and Flash Attention.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="qwen3_30b_a3b_moe")
    parser.add_argument("--hidden-size", type=positive_int)
    parser.add_argument("--ffn-hidden-size", type=positive_int)
    parser.add_argument("--moe-ffn-hidden-size", type=positive_int)
    parser.add_argument("--num-attention-heads", type=positive_int)
    parser.add_argument("--num-key-value-heads", type=positive_int)
    parser.add_argument("--head-dim", type=positive_int)
    parser.add_argument("--seq-length", type=positive_int)
    parser.add_argument("--micro-batch-size", type=positive_int)
    parser.add_argument("--vocab-size", type=positive_int)
    parser.add_argument("--tp-size", type=positive_int, default=1)
    parser.add_argument("--cp-size", type=positive_int, default=1)
    parser.add_argument("--ep-size", type=positive_int, default=1)
    parser.add_argument("--etp-size", type=positive_int)
    parser.add_argument("--num-experts", type=positive_int)
    parser.add_argument("--topk", type=positive_int)
    parser.add_argument("--num-shared-experts", type=non_negative_int)
    parser.add_argument("--shared-expert-ffn-hidden-size", type=positive_int)
    parser.add_argument("--moe-grouped-gemm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-local-experts", type=positive_int)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=positive_int, default=20)
    parser.add_argument(
        "--ops",
        type=comma_list,
        default=["all"],
        help="Comma separated selectors: all, torch, te, fa, moe, loss, optimizer, or an exact operator name.",
    )
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--attention-kv-context",
        choices=("local", "global"),
        default="local",
        help="Use local CP KV chunk or full global KV context for attention kernels.",
    )
    parser.add_argument("--optimizer-numel", type=positive_int, default=16_777_216)
    parser.add_argument("--vision-seq-length", type=positive_int)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Print the operator plan without importing torch.")
    parser.add_argument("--list-ops", action="store_true", help="List benchmarked operators without importing torch.")
    return parser.parse_args()


def finalize_config(config: dict) -> dict[str, int | float | str | bool | list[str] | Path | None]:
    ops = config.get("ops", ["all"])
    if isinstance(ops, str):
        config["ops"] = comma_list(ops)
    elif ops is None:
        config["ops"] = ["all"]
    else:
        config["ops"] = list(ops)
    config["head_dim"] = int(config.get("head_dim") or (int(config["hidden_size"]) // int(config["num_attention_heads"])))
    cp = int(config["cp_size"])
    seq = int(config["seq_length"])
    config["local_seq_length"] = div_ceil(seq, cp)
    config["attention_kv_seq_length"] = (
        seq if config["attention_kv_context"] == "global" else config["local_seq_length"]
    )
    config["tokens"] = int(config["micro_batch_size"]) * int(config["local_seq_length"])
    config["etp_size"] = int(config.get("etp_size") or config["tp_size"])
    config["has_qk_norm"] = bool(config.get("has_qk_norm", False))
    config["uses_mla"] = bool(config.get("uses_mla", False))
    config["has_dsa"] = bool(config.get("has_dsa", False))
    config["has_gated_delta_net"] = bool(config.get("has_gated_delta_net", False))
    config["has_vision"] = bool(config.get("has_vision", False))
    config["has_mtp"] = bool(config.get("has_mtp", False))
    config["has_dense_mlp"] = bool(
        config.get("has_dense_mlp", int(config["num_experts"]) <= 1)
    )
    config.setdefault("linear_conv_kernel_dim", 0)
    config.setdefault("linear_key_head_dim", 0)
    config.setdefault("linear_value_head_dim", 0)
    config.setdefault("linear_num_key_heads", 0)
    config.setdefault("linear_num_value_heads", 0)
    config.setdefault("partial_rotary_factor", 1.0)
    config.setdefault("q_lora_rank", 0)
    config.setdefault("kv_lora_rank", 0)
    config.setdefault("qk_nope_head_dim", 0)
    config.setdefault("qk_rope_head_dim", 0)
    config.setdefault("v_head_dim", config["head_dim"])
    config.setdefault("index_n_heads", 0)
    config.setdefault("index_head_dim", 0)
    config.setdefault("index_topk", 0)
    config.setdefault("sequence_parallel", True)
    config.setdefault("vision_seq_length", 0)
    config.setdefault("vision_num_images", 1)
    config.setdefault("vision_deepstack_count", 0)
    return config


def merged_config(args: argparse.Namespace) -> dict[str, int | float | str | bool | list[str] | Path | None]:
    config = dict(PRESETS[args.preset])
    for key, value in vars(args).items():
        if value is not None:
            config[key.replace("-", "_")] = value
    return finalize_config(config)


def build_benchmark_config(simulation_config, overrides: dict | None = None) -> dict:
    """Build benchmark configuration from the simulation configuration."""
    model = simulation_config.model
    transformer = simulation_config.transformer
    parallel = simulation_config.parallel

    config = {
        "hidden_size": transformer.hidden_size,
        "ffn_hidden_size": transformer.ffn_hidden_size,
        "moe_ffn_hidden_size": transformer.moe_ffn_hidden_size or transformer.ffn_hidden_size,
        "num_attention_heads": transformer.num_attention_heads,
        "num_key_value_heads": transformer.num_query_groups,
        "head_dim": transformer.kv_channels,
        "uses_mla": spec_uses_mla(simulation_config.model_spec),
        "q_lora_rank": transformer.q_lora_rank or 0,
        "kv_lora_rank": transformer.kv_lora_rank or 0,
        "qk_nope_head_dim": transformer.qk_nope_head_dim or 0,
        "qk_rope_head_dim": transformer.qk_rope_head_dim or 0,
        "v_head_dim": transformer.v_head_dim or transformer.kv_channels,
        "has_dsa": spec_uses_dsa(simulation_config.model_spec),
        "index_n_heads": transformer.index_n_heads,
        "index_head_dim": transformer.index_head_dim,
        "index_topk": transformer.index_topk,
        "seq_length": parallel.seq_length,
        "micro_batch_size": parallel.micro_batch_size,
        "vocab_size": model.vocab_size,
        "num_experts": transformer.num_moe_experts or 1,
        "topk": transformer.moe_router_topk or 1,
        "num_shared_experts": transformer.num_shared_experts or 0,
        "shared_expert_ffn_hidden_size": (
            transformer.shared_expert_ffn_hidden_size
            or transformer.moe_ffn_hidden_size
            or transformer.ffn_hidden_size
        ),
        "tp_size": parallel.tp_size,
        "cp_size": parallel.cp_size,
        "ep_size": parallel.ep_size,
        "etp_size": parallel.etp_size,
        "sequence_parallel": getattr(parallel, "sequence_parallel", True),
        "moe_grouped_gemm": getattr(parallel, "moe_grouped_gemm", True),
        "dtype": getattr(parallel, "benchmark_dtype", "bf16"),
        "device": getattr(parallel, "benchmark_device", "cuda"),
        "allow_cpu": getattr(parallel, "benchmark_allow_cpu", False),
        "warmup": getattr(parallel, "benchmark_warmup", 5),
        "iters": getattr(parallel, "benchmark_iters", 20),
        "ops": getattr(parallel, "benchmark_ops", ["all"]),
        "dropout_p": getattr(parallel, "benchmark_dropout_p", 0.0),
        "causal": getattr(parallel, "benchmark_causal", True),
        "attention_kv_context": getattr(parallel, "benchmark_attention_kv_context", "local"),
        "optimizer_numel": getattr(parallel, "benchmark_optimizer_numel", 16_777_216),
        "max_local_experts": getattr(parallel, "benchmark_max_local_experts", None),
        "has_qk_norm": spec_has_qk_norm(simulation_config.model_spec),
        "has_gated_delta_net": spec_uses_gated_delta_net(simulation_config.model_spec),
        "linear_conv_kernel_dim": transformer.linear_conv_kernel_dim,
        "linear_key_head_dim": transformer.linear_key_head_dim,
        "linear_value_head_dim": transformer.linear_value_head_dim,
        "linear_num_key_heads": transformer.linear_num_key_heads,
        "linear_num_value_heads": transformer.linear_num_value_heads,
        "partial_rotary_factor": transformer.partial_rotary_factor,
        "has_vision": spec_uses_vision(simulation_config.model_spec),
        "has_mtp": spec_uses_mtp(simulation_config.model_spec),
        "has_dense_mlp": any(
            not layer_uses_moe(transformer, layer_index)
            for layer_index in range(transformer.num_layers)
        ),
        "vision_hidden_size": simulation_config.vision.hidden_size,
        "vision_ffn_hidden_size": simulation_config.vision.ffn_hidden_size,
        "vision_num_heads": simulation_config.vision.num_attention_heads,
        "vision_seq_length": parallel.vision_seq_length or simulation_config.vision.num_position_embeddings,
        "vision_patch_size": simulation_config.vision.patch_size,
        "vision_temporal_patch_size": simulation_config.vision.temporal_patch_size,
        "vision_in_channels": simulation_config.vision.in_channels,
        "vision_spatial_merge_size": simulation_config.vision.spatial_merge_size,
        "vision_output_hidden_size": simulation_config.vision.output_hidden_size,
        "vision_num_images": parallel.vision_num_images,
        "vision_deepstack_count": len(
            simulation_config.vision.deepstack_visual_indexes
        ),
    }
    if overrides:
        config.update(overrides)
    return finalize_config(config)


def validate_config(config: dict) -> None:
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    tp = int(config["tp_size"])
    cp = int(config["cp_size"])
    ffn = int(config["ffn_hidden_size"])
    moe_ffn = int(config["moe_ffn_hidden_size"])
    shared_ffn = int(config["shared_expert_ffn_hidden_size"])
    etp = int(config["etp_size"])
    if heads % tp != 0:
        raise ValueError("--num-attention-heads must be divisible by --tp-size")
    if not config.get("uses_mla") and kv_heads % tp != 0:
        raise ValueError("--num-key-value-heads must be divisible by --tp-size")
    if hidden % tp != 0:
        raise ValueError("--hidden-size must be divisible by --tp-size")
    if ffn % tp != 0:
        raise ValueError("--ffn-hidden-size must be divisible by --tp-size")
    if moe_ffn % etp != 0:
        raise ValueError("--moe-ffn-hidden-size must be divisible by --etp-size")
    if shared_ffn % tp != 0:
        raise ValueError("--shared-expert-ffn-hidden-size must be divisible by --tp-size")
    if int(config["seq_length"]) % cp != 0:
        raise ValueError("--seq-length must be divisible by --cp-size for deterministic local attention shapes")
    if int(config["num_experts"]) % int(config["ep_size"]) != 0:
        raise ValueError("--num-experts must be divisible by --ep-size")
    if config.get("has_gated_delta_net"):
        if int(config["linear_num_key_heads"]) % tp != 0:
            raise ValueError("linear_num_key_heads must be divisible by --tp-size")
        if int(config["linear_num_value_heads"]) % tp != 0:
            raise ValueError("linear_num_value_heads must be divisible by --tp-size")
    if config.get("has_vision"):
        if int(config["vision_hidden_size"]) % tp != 0:
            raise ValueError("vision_hidden_size must be divisible by --tp-size")
        if int(config["vision_num_heads"]) % tp != 0:
            raise ValueError("vision_num_heads must be divisible by --tp-size")
        if int(config["vision_ffn_hidden_size"]) % tp != 0:
            raise ValueError("vision_ffn_hidden_size must be divisible by --tp-size")


def div_ceil(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def moe_tokens_per_local_expert(config: dict) -> int:
    tokens = int(config["tokens"])
    topk = int(config["topk"])
    experts = int(config["num_experts"])
    ep = int(config["ep_size"])
    tp = int(config["tp_size"])
    sequence_shard = tp if bool(config.get("sequence_parallel", True)) else 1
    return max(1, div_ceil(tokens * topk * ep, experts * sequence_shard))


def _extend_registered_plans(plan, config):
    for builder in registered_plan_builders():
        plan.extend(builder(config, OperatorPlan))
    return plan


def build_operator_plan(config: dict) -> list[OperatorPlan]:
    hidden = int(config["hidden_size"])
    ffn = int(config["ffn_hidden_size"])
    moe_ffn = int(config["moe_ffn_hidden_size"])
    seq = int(config["local_seq_length"])
    kv_seq = int(config["attention_kv_seq_length"])
    batch = int(config["micro_batch_size"])
    tokens = int(config["tokens"])
    tp = int(config["tp_size"])
    ep = int(config["ep_size"])
    etp = int(config["etp_size"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    vocab = int(config["vocab_size"])
    experts = int(config["num_experts"])
    topk = int(config["topk"])
    shared_experts = int(config["num_shared_experts"])
    shared_ffn = int(config["shared_expert_ffn_hidden_size"])
    local_experts = max(1, div_ceil(experts, ep))
    benchmark_local_experts = min(int(config.get("max_local_experts") or local_experts), local_experts)
    tokens_per_expert = moe_tokens_per_local_expert(config)
    uses_mla = bool(config.get("uses_mla", False))
    qk_head_dim = (
        int(config["qk_nope_head_dim"]) + int(config["qk_rope_head_dim"])
        if uses_mla
        else head_dim
    )
    value_head_dim = int(config.get("v_head_dim") or head_dim)
    q_heads_rank = max(1, heads // tp)
    kv_heads_rank = max(1, kv_heads // tp)
    attention_kv_heads_rank = q_heads_rank if uses_mla else kv_heads_rank
    qkv_out = (q_heads_rank + 2 * kv_heads_rank) * head_dim
    ffn_rank = max(1, ffn // tp)
    moe_ffn_rank = max(1, moe_ffn // etp)
    shared_ffn_rank = max(1, shared_ffn // tp)
    vocab_rank = max(1, vocab // tp)

    plan = [
        OperatorPlan("te_linear_qkv", "te", "linear_qkv -> te.pytorch.LayerNormLinear", f"[{tokens},{hidden}] x [{hidden},{qkv_out}]", "transformer_engine"),
        OperatorPlan("te_linear_proj", "te", "linear_proj -> te.pytorch.Linear", f"[{tokens},{q_heads_rank * head_dim}] x [{q_heads_rank * head_dim},{hidden}]", "transformer_engine"),
        OperatorPlan("te_rmsnorm", "te", "TENorm -> te.pytorch.RMSNorm", f"[{batch},{seq},{hidden}]", "transformer_engine"),
        OperatorPlan("torch_lm_head", "loss", "lm_head -> torch.nn.Module", f"[{tokens},{hidden}] x [{hidden},{vocab_rank}]"),
        OperatorPlan("torch_layernorm", "torch", "LayerNorm forward+backward", f"[{batch},{seq},{hidden}]"),
        OperatorPlan("torch_rmsnorm", "torch", "RMSNorm forward+backward", f"[{batch},{seq},{hidden}]"),
        OperatorPlan("te_fused_rope", "te", "Transformer Engine fused RoPE forward+backward", f"q/k=[{batch},{seq},{q_heads_rank},{head_dim}]", "transformer_engine"),
        OperatorPlan("torch_rope", "torch", "Fallback Rotary position embedding forward+backward", f"q/k=[{batch},{seq},{q_heads_rank},{head_dim}]"),
        OperatorPlan("torch_softmax_dropout", "torch", "Attention score softmax/dropout forward+backward", f"[{batch},{q_heads_rank},{seq},{kv_seq}]"),
        OperatorPlan("torch_bias_dropout_add", "torch", "Bias/dropout/residual add forward+backward", f"[{batch},{seq},{hidden}]"),
        OperatorPlan("torch_residual_add", "torch", "Residual add forward+backward without dropout", f"[{batch},{seq},{hidden}]"),
        OperatorPlan("torch_sdpa_attention", "torch", "PyTorch scaled_dot_product_attention forward+backward", f"q=[{batch},{q_heads_rank},{seq},{qk_head_dim}], k=[{batch},{attention_kv_heads_rank},{kv_seq},{qk_head_dim}], v=[{batch},{attention_kv_heads_rank},{kv_seq},{value_head_dim}]"),
        OperatorPlan("flash_attention", "fa", "Flash Attention forward+backward", f"q=[{batch},{seq},{q_heads_rank},{qk_head_dim}], k=[{batch},{kv_seq},{attention_kv_heads_rank},{qk_head_dim}], v=[{batch},{kv_seq},{attention_kv_heads_rank},{value_head_dim}]", "flash_attn"),
        OperatorPlan("torch_vocab_parallel_cross_entropy", "loss", "Vocab-parallel cross entropy approximation", f"logits=[{tokens},{vocab_rank}], tp={tp}"),
        OperatorPlan("torch_cross_entropy", "loss", "Fallback cross entropy over partitioned LM-head logits", f"logits=[{tokens},{vocab_rank}]"),
        OperatorPlan("torch_adamw_step", "optimizer", "AdamW optimizer update", f"numel={int(config['optimizer_numel'])}"),
    ]
    if config.get("has_mtp"):
        plan.append(
            OperatorPlan(
                "te_mtp_fc",
                "mtp",
                "MTP replicated 2H-to-H fusion projection",
                f"[{tokens},{2 * hidden}] x [{2 * hidden},{hidden // tp}]",
                "transformer_engine",
            )
        )
    if config.get("has_qk_norm"):
        plan.insert(
            6,
            OperatorPlan("torch_qk_rmsnorm", "torch", "Qwen3 Q/K RMSNorm forward+backward", f"q=[{batch},{seq},{q_heads_rank},{head_dim}], k=[{batch},{seq},{kv_heads_rank},{head_dim}]"),
        )
    if experts <= 1 or config.get("has_dense_mlp"):
        plan.extend(
            [
                OperatorPlan("te_dense_linear_fc1", "dense", "linear_fc1 -> te.pytorch.LayerNormLinear", f"[{tokens},{hidden}] x [{hidden},{2 * ffn_rank}]", "transformer_engine"),
                OperatorPlan("torch_swiglu_activation", "dense", "SwiGLU activation forward+backward", f"[{tokens},{2 * ffn_rank}]"),
                OperatorPlan("te_dense_linear_fc2", "dense", "linear_fc2 -> te.pytorch.Linear", f"[{tokens},{ffn_rank}] x [{ffn_rank},{hidden}]", "transformer_engine"),
            ]
        )
    if experts <= 1:
        return _extend_registered_plans(plan, config)

    plan.append(OperatorPlan("torch_moe_router", "moe", "router -> Router(MegatronModule): linear, softmax, top-k", f"[{tokens},{hidden}] x [{hidden},{experts}], topk={topk}"))
    if shared_experts:
        plan.extend(
            [
                OperatorPlan("te_shared_experts_linear_fc1", "moe", "shared_experts linear_fc1 -> te.pytorch.Linear", f"[{tokens},{hidden}] x [{hidden},{2 * shared_ffn_rank}], shared_experts={shared_experts}", "transformer_engine"),
                OperatorPlan("torch_shared_swiglu_activation", "moe", "Shared expert SwiGLU activation forward+backward", f"[{tokens},{2 * shared_ffn_rank}]"),
                OperatorPlan("te_shared_experts_linear_fc2", "moe", "shared_experts linear_fc2 -> te.pytorch.Linear", f"[{tokens},{shared_ffn_rank}] x [{shared_ffn_rank},{hidden}], shared_experts={shared_experts}", "transformer_engine"),
                OperatorPlan("te_shared_expert_gate", "moe", "Shared expert sigmoid gate", f"[{tokens},{hidden}] x [{hidden},1]", "transformer_engine"),
            ]
        )
    if config["moe_grouped_gemm"]:
        plan.extend(
            [
                OperatorPlan("te_moe_grouped_linear_fc1", "moe", "linear_fc1 -> te.pytorch.GroupedLinear", f"num_gemms={benchmark_local_experts}/{local_experts}, each [{tokens_per_expert},{hidden}] x [{hidden},{2 * moe_ffn_rank}], m_splits=[{tokens_per_expert}]x{benchmark_local_experts}", "transformer_engine"),
                OperatorPlan("torch_moe_swiglu_activation", "moe", "MoE SwiGLU activation forward+backward", f"[{benchmark_local_experts * tokens_per_expert},{2 * moe_ffn_rank}]"),
                OperatorPlan("te_moe_grouped_linear_fc2", "moe", "linear_fc2 -> te.pytorch.GroupedLinear", f"num_gemms={benchmark_local_experts}/{local_experts}, each [{tokens_per_expert},{moe_ffn_rank}] x [{moe_ffn_rank},{hidden}], m_splits=[{tokens_per_expert}]x{benchmark_local_experts}", "transformer_engine"),
            ]
        )
    else:
        plan.extend(
            [
                OperatorPlan("te_moe_linear_fc1", "moe", "linear_fc1 -> te.pytorch.Linear loop", f"calls={benchmark_local_experts}/{local_experts}, each [{tokens_per_expert},{hidden}] x [{hidden},{2 * moe_ffn_rank}]", "transformer_engine"),
                OperatorPlan("torch_moe_swiglu_activation", "moe", "MoE SwiGLU activation forward+backward", f"[{benchmark_local_experts * tokens_per_expert},{2 * moe_ffn_rank}]"),
                OperatorPlan("te_moe_linear_fc2", "moe", "linear_fc2 -> te.pytorch.Linear loop", f"calls={benchmark_local_experts}/{local_experts}, each [{tokens_per_expert},{moe_ffn_rank}] x [{moe_ffn_rank},{hidden}]", "transformer_engine"),
            ]
        )
    return _extend_registered_plans(plan, config)


def selected(plan: OperatorPlan, selectors: Iterable[str]) -> bool:
    selectors = set(selectors)
    return (
        "all" in selectors
        or plan.group in selectors
        or plan.name in selectors
        or (plan.group == "fa" and "flash" in selectors)
        or (plan.group == "te" and "transformer_engine" in selectors)
    )


def filtered_plan(config: dict) -> list[OperatorPlan]:
    plan = build_operator_plan(config)
    return [item for item in plan if selected(item, config["ops"])]


class BenchRunner:
    def __init__(self, config: dict):
        import torch

        self.torch = torch
        self.config = config
        requested = str(config["device"])
        if requested == "cuda" and not torch.cuda.is_available():
            if not config["allow_cpu"]:
                raise RuntimeError("CUDA/HIP device is not available. Use --allow-cpu for functional CPU smoke tests.")
            requested = "cpu"
        self.device = torch.device(requested)
        self.dtype = self._dtype(str(config["dtype"]))
        self.warmup = int(config["warmup"])
        self.iters = int(config["iters"])
        torch.manual_seed(1234)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(1234)
            torch.backends.cuda.matmul.allow_tf32 = True

    def _dtype(self, name: str):
        if name == "fp16":
            return self.torch.float16
        if name == "bf16":
            return self.torch.bfloat16
        return self.torch.float32

    def _randn(self, *shape: int, requires_grad: bool = False):
        return self.torch.randn(*shape, device=self.device, dtype=self.dtype, requires_grad=requires_grad)

    def _zeros_grad(self, *tensors):
        for tensor in tensors:
            tensor.grad = None

    def _sync(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def _measure(
        self,
        plan: OperatorPlan,
        fn: Callable[[], None],
        flops_per_iter: float | None = None,
    ) -> BenchmarkResult:
        samples: list[float] = []
        """
        执行传入的函数并返回测试结果
        """
        try:
            for _ in range(max(0, self.warmup)):
                fn()
            self._sync()  # CPU 会阻塞等待，直到 GPU 完成所有已提交的操作，可用于精确测量GPU时间
            if self.device.type == "cuda":
                self.torch.cuda.reset_peak_memory_stats(self.device)
                for _ in range(self.iters):
                    start = self.torch.cuda.Event(enable_timing=True)
                    end = self.torch.cuda.Event(enable_timing=True)
                    start.record()
                    fn()
                    end.record()
                    end.synchronize()
                    samples.append(float(start.elapsed_time(end)))
                peak_mb = self.torch.cuda.max_memory_allocated(self.device) / 1024**2
            else:
                peak_mb = None
                for _ in range(self.iters):
                    start = time.perf_counter()
                    fn()
                    samples.append((time.perf_counter() - start) * 1000)
            samples.sort()
            mean_ms = sum(samples) / len(samples)
            tflops = None
            if flops_per_iter and mean_ms > 0:
                tflops = flops_per_iter / (mean_ms / 1000) / 1e12
            return BenchmarkResult(
                name=plan.name,
                group=plan.group,
                status="ok",
                mean_ms=mean_ms,
                p50_ms=samples[len(samples) // 2],
                min_ms=samples[0],
                max_ms=samples[-1],
                tflops=tflops,
                peak_memory_mb=peak_mb,
                shape=plan.shape,
            )
        except Exception as exc:
            return BenchmarkResult(plan.name, plan.group, "skipped", None, None, None, None, None, None, plan.shape, str(exc))

    def _linear(self, plan: OperatorPlan, in_features: int, out_features: int, tokens: int) -> BenchmarkResult:
        x = self._randn(tokens, in_features, requires_grad=True)
        weight = self._randn(in_features, out_features, requires_grad=True)

        def fn():
            y = x.matmul(weight)
            y.float().sum().backward()
            self._zeros_grad(x, weight)

        return self._measure(plan, fn, flops_per_iter=6 * tokens * in_features * out_features)

    def _torch_linear_module(self, plan: OperatorPlan, in_features: int, out_features: int, tokens: int) -> BenchmarkResult:
        module = self.torch.nn.Linear(in_features, out_features, bias=False, device=self.device, dtype=self.dtype)
        x = self._randn(tokens, in_features, requires_grad=True)
        return self._module_bwd(plan, module, x, flops_per_iter=6 * tokens * in_features * out_features)

    def _module_bwd(self, plan: OperatorPlan, module, x, flops_per_iter: float | None = None) -> BenchmarkResult:
        module.train()

        def fn():
            y = module(x)
            if isinstance(y, tuple):
                y = y[0]
            y.float().sum().backward()
            module.zero_grad(set_to_none=True)
            x.grad = None

        return self._measure(plan, fn, flops_per_iter=flops_per_iter)

    def run_plan(self, plan: OperatorPlan) -> BenchmarkResult:
        registered_case = get_benchmark_case(plan.name)
        if registered_case is not None:
            return registered_case(self, plan)
        cfg = self.config
        hidden = int(cfg["hidden_size"])
        ffn = int(cfg["ffn_hidden_size"])
        moe_ffn = int(cfg["moe_ffn_hidden_size"])
        seq = int(cfg["local_seq_length"])
        kv_seq = int(cfg["attention_kv_seq_length"])
        batch = int(cfg["micro_batch_size"])
        tokens = int(cfg["tokens"])
        tp = int(cfg["tp_size"])
        ep = int(cfg["ep_size"])
        etp = int(cfg["etp_size"])
        heads = int(cfg["num_attention_heads"])
        kv_heads = int(cfg["num_key_value_heads"])
        head_dim = int(cfg["head_dim"])
        experts = int(cfg["num_experts"])
        topk = int(cfg["topk"])
        vocab = int(cfg["vocab_size"])
        shared_ffn = int(cfg["shared_expert_ffn_hidden_size"])
        q_heads_rank = max(1, heads // tp)
        kv_heads_rank = max(1, kv_heads // tp)
        qkv_out = (q_heads_rank + 2 * kv_heads_rank) * head_dim
        ffn_rank = max(1, ffn // tp)
        moe_ffn_rank = max(1, moe_ffn // etp)
        shared_ffn_rank = max(1, shared_ffn // tp)
        vocab_rank = max(1, vocab // tp)

        if plan.name == "torch_lm_head":
            return self._torch_linear_module(plan, hidden, vocab_rank, tokens)
        if plan.name in {
            "torch_swiglu_activation",
            "torch_moe_swiglu_activation",
            "torch_shared_swiglu_activation",
        }:
            activation_ffn = {
                "torch_swiglu_activation": ffn_rank,
                "torch_moe_swiglu_activation": moe_ffn_rank,
                "torch_shared_swiglu_activation": shared_ffn_rank,
            }[plan.name]
            activation_tokens = tokens
            if plan.name == "torch_moe_swiglu_activation":
                local_experts = max(1, div_ceil(experts, ep))
                benchmark_local_experts = min(int(cfg.get("max_local_experts") or local_experts), local_experts)
                activation_tokens = benchmark_local_experts * max(1, tokens * topk // experts)
            x = self._randn(activation_tokens, 2 * activation_ffn, requires_grad=True)

            def fn():
                gate, up = x.chunk(2, dim=-1)
                y = self.torch.nn.functional.silu(gate) * up
                y.float().sum().backward()
                x.grad = None

            return self._measure(plan, fn)
        if plan.name == "torch_layernorm":
            module = self.torch.nn.LayerNorm(hidden, device=self.device, dtype=self.dtype)
            x = self._randn(batch, seq, hidden, requires_grad=True)
            return self._module_bwd(plan, module, x)
        if plan.name == "torch_rmsnorm":
            x = self._randn(batch, seq, hidden, requires_grad=True)
            weight = self.torch.ones(hidden, device=self.device, dtype=self.dtype, requires_grad=True)

            def fn():
                y = x * self.torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6).to(self.dtype)
                y = y * weight
                y.float().sum().backward()
                self._zeros_grad(x, weight)

            return self._measure(plan, fn)
        if plan.name == "torch_qk_rmsnorm":
            q = self._randn(batch, seq, q_heads_rank, head_dim, requires_grad=True)
            k = self._randn(batch, seq, kv_heads_rank, head_dim, requires_grad=True)
            q_weight = self.torch.ones(head_dim, device=self.device, dtype=self.dtype, requires_grad=True)
            k_weight = self.torch.ones(head_dim, device=self.device, dtype=self.dtype, requires_grad=True)

            def rms_norm(x, weight):
                if hasattr(self.torch.nn.functional, "rms_norm"):
                    return self.torch.nn.functional.rms_norm(x, (head_dim,), weight=weight, eps=1e-6)
                y = x * self.torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6).to(self.dtype)
                return y * weight

            def fn():
                y = rms_norm(q, q_weight).sum() + rms_norm(k, k_weight).sum()
                y.float().backward()
                self._zeros_grad(q, k, q_weight, k_weight)

            return self._measure(plan, fn)
        if plan.name == "te_fused_rope":
            return self._run_te_fused_rope(plan, batch, seq, q_heads_rank, kv_heads_rank, head_dim)
        if plan.name == "torch_rope":
            q = self._randn(batch, seq, q_heads_rank, head_dim, requires_grad=True)
            k = self._randn(batch, seq, kv_heads_rank, head_dim, requires_grad=True)
            freqs = self.torch.arange(0, head_dim, 2, device=self.device, dtype=self.torch.float32) / head_dim
            pos = self.torch.arange(seq, device=self.device, dtype=self.torch.float32)
            angles = pos[:, None] * (10000 ** -freqs[None, :])
            cos = angles.cos()[None, :, None, :]
            sin = angles.sin()[None, :, None, :]

            def apply_rope(x):
                even = x[..., 0::2].float()
                odd = x[..., 1::2].float()
                out = self.torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1)
                return out.flatten(-2).to(self.dtype)

            def fn():
                y = apply_rope(q).sum() + apply_rope(k).sum()
                y.float().backward()
                self._zeros_grad(q, k)

            return self._measure(plan, fn)
        if plan.name == "torch_softmax_dropout":
            scores = self._randn(batch, q_heads_rank, seq, kv_seq, requires_grad=True)
            dropout_p = float(cfg["dropout_p"])

            def fn():
                y = self.torch.softmax(scores.float(), dim=-1).to(self.dtype)
                if dropout_p:
                    y = self.torch.nn.functional.dropout(y, p=dropout_p, training=True)
                y.float().sum().backward()
                scores.grad = None

            return self._measure(plan, fn)
        if plan.name == "torch_bias_dropout_add":
            x = self._randn(batch, seq, hidden, requires_grad=True)
            residual = self._randn(batch, seq, hidden, requires_grad=True)
            bias = self._randn(hidden, requires_grad=True)
            dropout_p = float(cfg["dropout_p"])

            def fn():
                y = x + bias
                if dropout_p:
                    y = self.torch.nn.functional.dropout(y, p=dropout_p, training=True)
                y = y + residual
                y.float().sum().backward()
                self._zeros_grad(x, residual, bias)

            return self._measure(plan, fn)
        if plan.name == "torch_residual_add":
            x = self._randn(batch, seq, hidden, requires_grad=True)
            residual = self._randn(batch, seq, hidden, requires_grad=True)

            def fn():
                y = x + residual
                y.float().sum().backward()
                self._zeros_grad(x, residual)

            return self._measure(plan, fn)
        if plan.name == "torch_sdpa_attention":
            q = self._randn(batch, q_heads_rank, seq, head_dim, requires_grad=True)
            k = self._randn(batch, kv_heads_rank, kv_seq, head_dim, requires_grad=True)
            v = self._randn(batch, kv_heads_rank, kv_seq, head_dim, requires_grad=True)

            def fn():
                kk, vv = k, v
                if q_heads_rank != kv_heads_rank:
                    repeat = max(1, q_heads_rank // kv_heads_rank)
                    kk = k.repeat_interleave(repeat, dim=1)
                    vv = v.repeat_interleave(repeat, dim=1)
                y = self.torch.nn.functional.scaled_dot_product_attention(
                    q, kk, vv, dropout_p=float(cfg["dropout_p"]), is_causal=bool(cfg["causal"])
                )
                y.float().sum().backward()
                self._zeros_grad(q, k, v)

            flops = 4 * batch * q_heads_rank * seq * kv_seq * head_dim
            return self._measure(plan, fn, flops_per_iter=flops)
        if plan.name == "flash_attention":
            flash_attn_func = self._import_flash_attention()
            uses_mla = bool(cfg.get("uses_mla", False))
            qk_head_dim = (
                int(cfg["qk_nope_head_dim"]) + int(cfg["qk_rope_head_dim"])
                if uses_mla
                else head_dim
            )
            value_head_dim = int(cfg.get("v_head_dim") or head_dim)
            attention_kv_heads_rank = q_heads_rank if uses_mla else kv_heads_rank
            q = self._randn(batch, seq, q_heads_rank, qk_head_dim, requires_grad=True)
            k = self._randn(batch, kv_seq, attention_kv_heads_rank, qk_head_dim, requires_grad=True)
            v = self._randn(batch, kv_seq, attention_kv_heads_rank, value_head_dim, requires_grad=True)

            def fn():
                y = flash_attn_func(q, k, v, dropout_p=float(cfg["dropout_p"]), causal=bool(cfg["causal"]))
                y.float().sum().backward()
                self._zeros_grad(q, k, v)

            flops = (
                    2
                    * batch
                    * q_heads_rank
                    * seq
                    * kv_seq
                    * (qk_head_dim + value_head_dim)
            )
            return self._measure(plan, fn, flops_per_iter=flops)
        if plan.name.startswith("te_"):
            return self._run_transformer_engine(
                plan,
                hidden=hidden,
                qkv_out=qkv_out,
                proj_in=q_heads_rank * head_dim,
                ffn_rank=ffn_rank,
                moe_ffn_rank=moe_ffn_rank,
                shared_ffn_rank=shared_ffn_rank,
                tokens=tokens,
                batch=batch,
                seq=seq,
                experts=experts,
                topk=topk,
                ep=ep,
            )
        if plan.name == "torch_moe_router":
            x = self._randn(tokens, hidden, requires_grad=True)
            gate = self._randn(hidden, experts, requires_grad=True)

            def fn():
                logits = x.matmul(gate).float()
                probs = self.torch.softmax(logits, dim=-1)
                vals, _ = self.torch.topk(probs, k=min(topk, experts), dim=-1)
                vals.sum().backward()
                self._zeros_grad(x, gate)

            return self._measure(plan, fn, flops_per_iter=6 * tokens * hidden * experts)
        if plan.name == "torch_vocab_parallel_cross_entropy":
            logits = self._randn(tokens, vocab_rank, requires_grad=True)
            targets = self.torch.randint(0, vocab_rank, (tokens,), device=self.device)

            def fn():
                logits_float = logits.float()
                local_max = logits_float.max(dim=-1, keepdim=True).values
                exp_logits = self.torch.exp(logits_float - local_max)
                local_sum = exp_logits.sum(dim=-1)
                target_logits = logits_float.gather(1, targets.view(-1, 1)).squeeze(1)
                loss = (self.torch.log(local_sum) + local_max.squeeze(1) - target_logits).mean()
                loss.backward()
                self._zeros_grad(logits)

            return self._measure(plan, fn)
        if plan.name == "torch_cross_entropy":
            logits = self._randn(tokens, vocab_rank, requires_grad=True)
            targets = self.torch.randint(0, vocab_rank, (tokens,), device=self.device)

            def fn():
                loss = self.torch.nn.functional.cross_entropy(logits.float(), targets)
                loss.backward()
                self._zeros_grad(logits)

            return self._measure(plan, fn)
        if plan.name == "torch_adamw_step":
            param = self._randn(int(cfg["optimizer_numel"]), requires_grad=True)
            optimizer = self.torch.optim.AdamW([param], lr=1e-4)

            def fn():
                param.grad = self.torch.ones_like(param)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            return self._measure(plan, fn)
        return BenchmarkResult(plan.name, plan.group, "skipped", None, None, None, None, None, None, plan.shape, "no runner registered")

    def _import_flash_attention(self):
        try:
            from flash_attn import flash_attn_func

            return flash_attn_func
        except Exception:
            from flash_attn_interface import flash_attn_func

            return flash_attn_func

    def _run_te_fused_rope(
        self,
        plan: OperatorPlan,
        batch: int,
        seq: int,
        q_heads_rank: int,
        kv_heads_rank: int,
        head_dim: int,
    ) -> BenchmarkResult:
        errors = []
        apply_fused_rope = None
        freqs = None

        def import_module(module_name):
            try:
                return importlib.import_module(module_name)
            except Exception as exc:
                errors.append(f"{module_name}: {exc}")
                return None

        def build_megatron_freqs():
            rotary_module = import_module("megatron.core.models.common.embeddings.rotary_pos_embedding")
            if rotary_module is None:
                return None
            rotary_cls = getattr(rotary_module, "RotaryEmbedding", None)
            if rotary_cls is None:
                errors.append("megatron RotaryEmbedding: missing")
                return None
            use_cpu_initialization = self.device.type == "cpu"
            attempts = (
                lambda: rotary_cls(
                    kv_channels=head_dim,
                    rotary_percent=1.0,
                    rotary_interleaved=False,
                    use_cpu_initialization=use_cpu_initialization,
                )(seq),
                lambda: rotary_cls(head_dim, 1.0, rotary_interleaved=False)(seq),
            )
            last_error = None
            for attempt in attempts:
                try:
                    freq = attempt()
                    return freq.to(device=self.device) if hasattr(freq, "to") else freq
                except Exception as exc:
                    last_error = exc
            errors.append(f"megatron RotaryEmbedding build: {last_error}")
            return None

        def build_te_freqs():
            rotary_cls = None
            for module_name in (
                "transformer_engine.pytorch.attention.rope",
                "transformer_engine.pytorch.attention",
                "transformer_engine.pytorch",
            ):
                module = import_module(module_name)
                if module is not None:
                    rotary_cls = rotary_cls or getattr(module, "RotaryPositionEmbedding", None)
            if rotary_cls is None:
                errors.append("TE RotaryPositionEmbedding: missing")
                return None
            attempts = (
                lambda: rotary_cls(head_dim)(seq),
                lambda: rotary_cls(head_dim, rotary_interleaved=False)(seq),
            )
            last_error = None
            for attempt in attempts:
                try:
                    freq = attempt()
                    return freq.to(device=self.device) if hasattr(freq, "to") else freq
                except Exception as exc:
                    last_error = exc
            errors.append(f"TE RotaryPositionEmbedding build: {last_error}")
            return None

        megatron_ext = import_module("megatron.core.extensions.transformer_engine")
        if megatron_ext is not None:
            fused_apply = getattr(megatron_ext, "fused_apply_rotary_pos_emb", None)
            freqs = build_megatron_freqs()
            if fused_apply is not None and freqs is not None:
                apply_fused_rope = lambda x: fused_apply(x, freqs, interleaved=False)

        if apply_fused_rope is None:
            rope_module = import_module("transformer_engine.pytorch.attention.rope")
            apply_rotary_pos_emb = getattr(rope_module, "apply_rotary_pos_emb", None) if rope_module else None
            freqs = freqs or build_te_freqs()
            if apply_rotary_pos_emb is not None and freqs is not None:
                def apply_te_rope(x):
                    attempts = (
                        {"tensor_format": "sbhd", "interleaved": False, "fused": True},
                        {"tensor_format": "sbhd", "fused": True},
                    )
                    last_error = None
                    for kwargs in attempts:
                        try:
                            return apply_rotary_pos_emb(x, freqs, **kwargs)
                        except TypeError as exc:
                            last_error = exc
                    raise last_error

                apply_fused_rope = apply_te_rope

        if apply_fused_rope is None:
            message = "missing TE RoPE API"
            if errors:
                message += "; " + " | ".join(errors[:6])
            return BenchmarkResult(plan.name, plan.group, "skipped", None, None, None, None, None, None, plan.shape, message)

        q = self._randn(seq, batch, q_heads_rank, head_dim, requires_grad=True)
        k = self._randn(seq, batch, kv_heads_rank, head_dim, requires_grad=True)

        def fn():
            q_out = apply_fused_rope(q)
            k_out = apply_fused_rope(k)
            if isinstance(q_out, tuple):
                q_out = q_out[0]
            if isinstance(k_out, tuple):
                k_out = k_out[0]
            y = q_out.float().sum() + k_out.float().sum()
            y.backward()
            self._zeros_grad(q, k)

        return self._measure(plan, fn)

    def _make_te_module(self, cls, *args, **kwargs):
        attempts = [
            {"params_dtype": self.dtype, "device": self.device},
            {"dtype": self.dtype, "device": self.device},
            {"params_dtype": self.dtype},
            {},
        ]
        last_error = None
        for extra in attempts:
            try:
                module = cls(*args, **kwargs, **extra)
                return module.to(device=self.device, dtype=self.dtype)
            except Exception as exc:
                last_error = exc
        raise last_error

    def _run_transformer_engine(
        self,
        plan: OperatorPlan,
        *,
        hidden: int,
        qkv_out: int,
        proj_in: int,
        ffn_rank: int,
        moe_ffn_rank: int,
        shared_ffn_rank: int,
        tokens: int,
        batch: int,
        seq: int,
        experts: int,
        topk: int,
        ep: int,
    ) -> BenchmarkResult:
        import transformer_engine.pytorch as te

        if plan.name == "te_linear_qkv":
            module = self._make_te_module(te.LayerNormLinear, hidden, qkv_out, bias=False, normalization="RMSNorm")
            x = self._randn(tokens, hidden, requires_grad=True)
            return self._module_bwd(plan, module, x, flops_per_iter=6 * tokens * hidden * qkv_out)
        if plan.name == "te_linear_proj":
            module = self._make_te_module(te.Linear, proj_in, hidden, bias=False)
            x = self._randn(tokens, proj_in, requires_grad=True)
            return self._module_bwd(plan, module, x, flops_per_iter=6 * tokens * proj_in * hidden)
        if plan.name == "te_rmsnorm":
            module = self._make_te_module(te.RMSNorm, hidden)
            x = self._randn(batch, seq, hidden, requires_grad=True)
            return self._module_bwd(plan, module, x)
        if plan.name == "te_shared_expert_gate":
            module = self._make_te_module(te.Linear, hidden, 1, bias=False)
            x = self._randn(tokens, hidden, requires_grad=True)
            return self._module_bwd(
                plan, module, x, flops_per_iter=6 * tokens * hidden
            )
        if plan.name == "te_dense_linear_fc1":
            module = self._make_te_module(te.LayerNormLinear, hidden, 2 * ffn_rank, bias=False, normalization="RMSNorm")
            x = self._randn(tokens, hidden, requires_grad=True)
            return self._module_bwd(plan, module, x, flops_per_iter=6 * tokens * hidden * 2 * ffn_rank)
        if plan.name == "te_dense_linear_fc2":
            module = self._make_te_module(te.Linear, ffn_rank, hidden, bias=False)
            x = self._randn(tokens, ffn_rank, requires_grad=True)
            return self._module_bwd(plan, module, x, flops_per_iter=6 * tokens * ffn_rank * hidden)
        if plan.name == "te_mtp_fc":
            tp = int(self.config["tp_size"])
            out_features = hidden // tp
            module = self._make_te_module(te.Linear, 2 * hidden, out_features, bias=False)
            x = self._randn(tokens, 2 * hidden, requires_grad=True)
            return self._module_bwd(
                plan,
                module,
                x,
                flops_per_iter=6 * tokens * 2 * hidden * out_features,
            )
        if plan.name == "te_shared_experts_linear_fc1":
            module = self._make_te_module(te.Linear, hidden, 2 * shared_ffn_rank, bias=False)
            x = self._randn(tokens, hidden, requires_grad=True)
            return self._module_bwd(plan, module, x, flops_per_iter=6 * tokens * hidden * 2 * shared_ffn_rank)
        if plan.name == "te_shared_experts_linear_fc2":
            module = self._make_te_module(te.Linear, shared_ffn_rank, hidden, bias=False)
            x = self._randn(tokens, shared_ffn_rank, requires_grad=True)
            return self._module_bwd(plan, module, x, flops_per_iter=6 * tokens * shared_ffn_rank * hidden)
        local_experts = max(1, div_ceil(experts, ep))
        local_experts = min(int(self.config.get("max_local_experts") or local_experts), local_experts)
        tokens_per_expert = moe_tokens_per_local_expert(self.config)
        if plan.name == "te_moe_grouped_linear_fc1":
            return self._te_grouped_linear(plan, te, local_experts, hidden, 2 * moe_ffn_rank, tokens_per_expert)
        if plan.name == "te_moe_grouped_linear_fc2":
            return self._te_grouped_linear(plan, te, local_experts, moe_ffn_rank, hidden, tokens_per_expert)
        if plan.name == "te_moe_linear_fc1":
            return self._te_linear_loop(plan, te, local_experts, hidden, 2 * moe_ffn_rank, tokens_per_expert)
        if plan.name == "te_moe_linear_fc2":
            return self._te_linear_loop(plan, te, local_experts, moe_ffn_rank, hidden, tokens_per_expert)
        raise RuntimeError(f"unsupported Transformer Engine plan: {plan.name}")

    def _te_grouped_linear(self, plan: OperatorPlan, te, num_gemms: int, in_features: int, out_features: int, tokens_per_expert: int) -> BenchmarkResult:
        module = self._make_te_module(te.GroupedLinear, num_gemms, in_features, out_features, bias=False)
        x = self._randn(num_gemms * tokens_per_expert, in_features, requires_grad=True)
        m_splits = [tokens_per_expert] * num_gemms

        def fn():
            y = module(x, m_splits)
            y.float().sum().backward()
            module.zero_grad(set_to_none=True)
            x.grad = None

        flops = 6 * num_gemms * tokens_per_expert * in_features * out_features
        return self._measure(plan, fn, flops_per_iter=flops)

    def _te_linear_loop(self, plan: OperatorPlan, te, num_gemms: int, in_features: int, out_features: int, tokens_per_expert: int) -> BenchmarkResult:
        modules = [self._make_te_module(te.Linear, in_features, out_features, bias=False) for _ in range(num_gemms)]
        xs = [self._randn(tokens_per_expert, in_features, requires_grad=True) for _ in range(num_gemms)]

        def fn():
            total = None
            for module, x in zip(modules, xs):
                y = module(x).float().sum()
                total = y if total is None else total + y
            total.backward()
            for module, x in zip(modules, xs):
                module.zero_grad(set_to_none=True)
                x.grad = None

        flops = 6 * num_gemms * tokens_per_expert * in_features * out_features
        return self._measure(plan, fn, flops_per_iter=flops)


def write_outputs(results: list[BenchmarkResult], output_json: Path | None, output_csv: Path | None) -> None:
    rows = [asdict(result) for result in results]
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [field.name for field in BenchmarkResult.__dataclass_fields__.values()])
            writer.writeheader()
            writer.writerows(rows)


def print_results(results: list[BenchmarkResult]) -> None:
    rows = []
    for item in results:
        rows.append(
            {
                "operator": item.name,
                "group": item.group,
                "status": item.status,
                "mean_ms": "-" if item.mean_ms is None else f"{item.mean_ms:.4f}",
                "tflops": "-" if item.tflops is None else f"{item.tflops:.2f}",
                "peak_mb": "-" if item.peak_memory_mb is None else f"{item.peak_memory_mb:.1f}",
                "shape": item.shape,
                "message": item.message or "",
            }
        )
    log_table(logger, "operator benchmark results", rows)


def train_env_valid():
    # 纯理论计算的环境不需要torch
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        # 不含torch
        logger.debug("torch is not present in the current virtual environment")
        return False


def _get_package_version(package_name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return str(version(package_name))
    except PackageNotFoundError:
        return None


def _get_driver_version() -> str | None:
    try:
        completed = subprocess.run(
            ["hy-smi", "--showdriverversion"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
    except Exception:
        return None
    match = re.search(r"Driver Version:\s*([^\r\n]+)", completed.stdout or "")
    return match.group(1).strip() if match else None


def get_env_info(device_index: int = 0) -> dict[str, Any]:
    """
    获取关键运行环境信息。

    Args:
        device_index: 需要查询的 GPU 编号，默认为 0。
    """
    import torch

    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0

    if cuda_available and not 0 <= device_index < gpu_count:
        raise ValueError(
            f"无效的 GPU 编号: {device_index}，当前可用 GPU 数量为 {gpu_count}"
        )

    gpu_name = None
    raw_gpu_name = None
    if gpu_count > 0:
        raw_gpu_name = torch.cuda.get_device_properties(device_index).name
        # Import lazily to keep --list-ops/--dry-run independent from profile IO.
        from hcu_train_simulator.benchmarks.profile import normalize_accelerator

        gpu_name = normalize_accelerator(raw_gpu_name) or raw_gpu_name

    return {
        "accelerator": gpu_name,
        "accelerator_raw": raw_gpu_name,
        "driver": _get_driver_version(),
        "torch_version": str(torch.__version__),
        "fa_version": _get_package_version("flash-attn"),
        "te_version": _get_package_version("transformer-engine"),
        "dtk_version": read_dtk_version(),
        "triton_version": _get_package_version("triton"),
        "cuda_available": cuda_available,
        "gpu_count": gpu_count,
    }


def run_benchmark(config: dict) -> list[BenchmarkResult]:
    validate_config(config)
    plan = filtered_plan(config)
    runner = BenchRunner(config)
    return [runner.run_plan(item) for item in plan]


def main():
    args = parse_args()
    config = merged_config(args)
    validate_config(config)
    plan = filtered_plan(config)

    if args.list_ops:
        rows = [
            {
                "operator": item.name,
                "group": item.group,
                "requires": item.requires,
                "description": item.description,
                "shape": item.shape,
            }
            for item in build_operator_plan(config)
        ]
        log_table(logger, "operator benchmark plan", rows)
        return 0, []
    if args.dry_run:
        logger.info(
            "operator benchmark dry run\n%s",
            json.dumps({"config": config, "operators": [asdict(item) for item in plan]}, indent=2, default=str),
        )
        return 0, []

    if not args.allow_cpu and not train_env_valid():
        logger.warning("no CUDA-capable PyTorch training environment is available for the actual operator benchmark")
        return 0, []

    results = run_benchmark(config)
    write_outputs(results, args.output_json, args.output_csv)
    print_results(results)
    run_state = 0 if all(item.status in {"ok", "skipped"} for item in results) else 1
    return run_state, results


if __name__ == "__main__":
    state, _ = main()
    raise SystemExit(state)
