# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek V3 MLA benchmark plans and Transformer Engine cases."""

from hcu_train_simulator.benchmarks.registry import (
    register_benchmark_case,
    register_benchmark_mapping,
    register_plan_builder,
)


register_benchmark_mapping(
    {
        "mla_q_down": "te_deepseek_mla_q_down",
        "mla_q_up": "te_deepseek_mla_q_up",
        "mla_kv_down": "te_deepseek_mla_kv_down",
        "mla_kv_up": "te_deepseek_mla_kv_up",
    },
    model_adapter="deepseek_v3",
)


def _mla_dims(config):
    tp = int(config["tp_size"])
    qk_dim = int(config["qk_nope_head_dim"]) + int(config["qk_rope_head_dim"])
    value_dim = int(config["v_head_dim"])
    q_heads_rank = int(config["num_attention_heads"]) // tp
    q_lora_rank = int(config.get("q_lora_rank") or 0)
    kv_lora_rank = int(config["kv_lora_rank"])
    return {
        "tokens": int(config["tokens"]),
        "hidden": int(config["hidden_size"]),
        "q_lora_rank": q_lora_rank,
        "kv_lora_rank": kv_lora_rank,
        "q_rope_dim": int(config["qk_rope_head_dim"]),
        "q_projection_rank": q_heads_rank * qk_dim,
        "kv_projection_rank": q_heads_rank
        * (int(config["qk_nope_head_dim"]) + value_dim),
    }


@register_plan_builder
def build_deepseek_mla_operator_plans(config, plan_type):
    if not config.get("uses_mla"):
        return []
    dims = _mla_dims(config)
    plans = []
    if dims["q_lora_rank"]:
        plans.append(
            plan_type(
                "te_deepseek_mla_q_down",
                "deepseek_mla",
                "DeepSeek MLA Q down projection",
                f"[{dims['tokens']},{dims['hidden']}] x "
                f"[{dims['hidden']},{dims['q_lora_rank']}]",
                "transformer_engine",
            )
        )
    plans.extend(
        [
            plan_type(
                "te_deepseek_mla_q_up",
                "deepseek_mla",
                "DeepSeek MLA fused Q RMSNorm and up projection",
                f"[{dims['tokens']},{dims['q_lora_rank'] or dims['hidden']}] x "
                f"[{dims['q_lora_rank'] or dims['hidden']},{dims['q_projection_rank']}]",
                "transformer_engine",
            ),
            plan_type(
                "te_deepseek_mla_kv_down",
                "deepseek_mla",
                "DeepSeek MLA KV down projection",
                f"[{dims['tokens']},{dims['hidden']}] x "
                f"[{dims['hidden']},{dims['kv_lora_rank'] + dims['q_rope_dim']}]",
                "transformer_engine",
            ),
            plan_type(
                "te_deepseek_mla_kv_up",
                "deepseek_mla",
                "DeepSeek MLA fused KV RMSNorm and up projection",
                f"[{dims['tokens']},{dims['kv_lora_rank']}] x "
                f"[{dims['kv_lora_rank']},{dims['kv_projection_rank']}]",
                "transformer_engine",
            ),
        ]
    )
    return plans


def run_deepseek_mla_linear(runner, plan):
    import transformer_engine.pytorch as te

    dims = _mla_dims(runner.config)
    shapes = {
        "te_deepseek_mla_q_down": (
            dims["hidden"],
            dims["q_lora_rank"],
        ),
        "te_deepseek_mla_q_up": (
            dims["q_lora_rank"] or dims["hidden"],
            dims["q_projection_rank"],
        ),
        "te_deepseek_mla_kv_down": (
            dims["hidden"],
            dims["kv_lora_rank"] + dims["q_rope_dim"],
        ),
        "te_deepseek_mla_kv_up": (
            dims["kv_lora_rank"],
            dims["kv_projection_rank"],
        ),
    }
    in_features, out_features = shapes[plan.name]
    fused_norm = plan.name in {
        "te_deepseek_mla_q_up",
        "te_deepseek_mla_kv_up",
    }
    module_type = te.LayerNormLinear if fused_norm else te.Linear
    kwargs = {"bias": False}
    if fused_norm:
        kwargs["normalization"] = "RMSNorm"
    module = runner._make_te_module(module_type, in_features, out_features, **kwargs)
    x = runner._randn(dims["tokens"], in_features, requires_grad=True)
    return runner._module_bwd(
        plan,
        module,
        x,
        flops_per_iter=6 * dims["tokens"] * in_features * out_features,
    )


for _case_name in (
    "te_deepseek_mla_q_down",
    "te_deepseek_mla_q_up",
    "te_deepseek_mla_kv_down",
    "te_deepseek_mla_kv_up",
):
    register_benchmark_case(_case_name)(run_deepseek_mla_linear)
