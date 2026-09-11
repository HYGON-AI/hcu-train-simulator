# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 measured-kernel mappings, plans, and case implementations."""

from hcu_train_simulator.benchmarks.registry import (
    register_benchmark_case,
    register_benchmark_mapping,
    register_benchmark_preset,
    register_plan_builder,
)


register_benchmark_preset(
    "qwen3_5_9b_vlm",
    {
        "hidden_size": 4096,
        "ffn_hidden_size": 12288,
        "moe_ffn_hidden_size": 12288,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "seq_length": 4096,
        "micro_batch_size": 1,
        "vocab_size": 248320,
        "num_experts": 1,
        "topk": 1,
        "num_shared_experts": 0,
        "shared_expert_ffn_hidden_size": 12288,
        "has_qk_norm": True,
        "has_gated_delta_net": True,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "partial_rotary_factor": 0.25,
        "has_vision": True,
        "has_mtp": True,
        "vision_hidden_size": 1152,
        "vision_ffn_hidden_size": 4304,
        "vision_num_heads": 16,
        "vision_seq_length": 2304,
        "vision_patch_size": 16,
        "vision_temporal_patch_size": 2,
        "vision_in_channels": 3,
        "vision_spatial_merge_size": 2,
        "vision_output_hidden_size": 4096,
        "vision_num_images": 1,
    },
)


register_benchmark_mapping(
    {
        "rope": "torch_qwen35_partial_rope",
        "qwen35_q_proj_gate": "te_qwen35_q_proj_gate",
        "qwen35_k_proj": "te_qwen35_k_proj",
        "qwen35_v_proj": "te_qwen35_v_proj",
        "attention_output_gate": "torch_attention_output_gate",
        "gdn_in_proj_qkv": "te_gdn_in_proj_qkv",
        "gdn_in_proj_z": "te_gdn_in_proj_z",
        "gdn_in_proj_a": "te_gdn_in_proj_a",
        "gdn_in_proj_b": "te_gdn_in_proj_b",
        "gdn_causal_conv1d": "torch_gdn_causal_conv1d",
        "gdn_gate_preprocess": "torch_gdn_gate_preprocess",
        "gdn_qk_l2norm": "torch_gdn_qk_l2norm",
        "gated_delta_rule": "fla_gated_delta_rule",
        "gdn_gated_rmsnorm": "torch_gdn_gated_rmsnorm",
        "gdn_out_proj": "te_gdn_out_proj",
    },
    model_adapter="qwen3_5",
)


@register_plan_builder
def build_qwen35_operator_plans(config, plan_type):
    plans = []
    hidden = int(config["hidden_size"])
    seq = int(config["local_seq_length"])
    batch = int(config["micro_batch_size"])
    tokens = int(config["tokens"])
    tp = int(config["tp_size"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    q_heads_rank = max(1, heads // tp)
    kv_heads_rank = max(1, kv_heads // tp)

    if config.get("has_gated_delta_net"):
        key_heads = int(config["linear_num_key_heads"])
        value_heads = int(config["linear_num_value_heads"])
        key_head_dim = int(config["linear_key_head_dim"])
        value_head_dim = int(config["linear_value_head_dim"])
        key_dim_rank = key_heads * key_head_dim // tp
        value_dim_rank = value_heads * value_head_dim // tp
        value_heads_rank = value_heads // tp
        rotary_dim = int(head_dim * float(config["partial_rotary_factor"]))
        plans.extend(
            [
                plan_type("te_qwen35_q_proj_gate", "qwen35", "Gated attention Q+gate projection", f"[{tokens},{hidden}] x [{hidden},{2 * q_heads_rank * head_dim}]", "transformer_engine"),
                plan_type("te_qwen35_k_proj", "qwen35", "Gated attention K projection", f"[{tokens},{hidden}] x [{hidden},{kv_heads_rank * head_dim}]", "transformer_engine"),
                plan_type("te_qwen35_v_proj", "qwen35", "Gated attention V projection", f"[{tokens},{hidden}] x [{hidden},{kv_heads_rank * head_dim}]", "transformer_engine"),
                plan_type("torch_attention_output_gate", "qwen35", "Gated attention sigmoid output gate forward+backward", f"[{tokens},{q_heads_rank * head_dim}]"),
                plan_type("torch_qwen35_partial_rope", "qwen35", "Qwen3.5 partial mRoPE forward+backward", f"q=[{batch},{seq},{q_heads_rank},{rotary_dim}], k=[{batch},{seq},{kv_heads_rank},{rotary_dim}]"),
                plan_type("te_gdn_in_proj_qkv", "gdn", "GDN fused QKV projection", f"[{tokens},{hidden}] x [{hidden},{2 * key_dim_rank + value_dim_rank}]", "transformer_engine"),
                plan_type("te_gdn_in_proj_z", "gdn", "GDN Z projection", f"[{tokens},{hidden}] x [{hidden},{value_dim_rank}]", "transformer_engine"),
                plan_type("te_gdn_in_proj_a", "gdn", "GDN decay projection", f"[{tokens},{hidden}] x [{hidden},{value_heads_rank}]", "transformer_engine"),
                plan_type("te_gdn_in_proj_b", "gdn", "GDN beta projection", f"[{tokens},{hidden}] x [{hidden},{value_heads_rank}]", "transformer_engine"),
                plan_type("torch_gdn_causal_conv1d", "gdn", "GDN depthwise causal Conv1d+SiLU forward+backward", f"[{batch},{2 * key_dim_rank + value_dim_rank},{seq}], kernel={config['linear_conv_kernel_dim']}"),
                plan_type("torch_gdn_gate_preprocess", "gdn", "GDN sigmoid/softplus gate preprocessing forward+backward", f"a/b=[{batch},{seq},{value_heads_rank}]"),
                plan_type("torch_gdn_qk_l2norm", "gdn", "GDN Q/K L2 normalization forward+backward", f"q/k=[{batch},{seq},{key_heads // tp},{key_head_dim}]"),
                plan_type("fla_gated_delta_rule", "gdn", "FLA chunk gated-delta-rule kernel forward+backward", f"q/k=[{batch},{seq},{value_heads_rank},{key_head_dim}], v=[{batch},{seq},{value_heads_rank},{value_head_dim}]", "fla"),
                plan_type("torch_gdn_gated_rmsnorm", "gdn", "GDN gated RMSNorm forward+backward", f"x/z=[{batch},{seq},{value_heads_rank},{value_head_dim}]"),
                plan_type("te_gdn_out_proj", "gdn", "GDN output projection", f"[{tokens},{value_dim_rank}] x [{value_dim_rank},{hidden}]", "transformer_engine"),
            ]
        )

    return plans


def _qwen35_dims(config):
    tp = int(config["tp_size"])
    return {
        "tp": tp,
        "batch": int(config["micro_batch_size"]),
        "seq": int(config["local_seq_length"]),
        "tokens": int(config["tokens"]),
        "hidden": int(config["hidden_size"]),
        "q_heads": int(config["num_attention_heads"]) // tp,
        "kv_heads": int(config["num_key_value_heads"]) // tp,
        "head_dim": int(config["head_dim"]),
        "key_heads": int(config["linear_num_key_heads"]) // tp,
        "value_heads": int(config["linear_num_value_heads"]) // tp,
        "key_head_dim": int(config["linear_key_head_dim"]),
        "value_head_dim": int(config["linear_value_head_dim"]),
    }


@register_benchmark_case("torch_qwen35_partial_rope")
def run_partial_rope(runner, plan):
    dims = _qwen35_dims(runner.config)
    rotary_dim = int(dims["head_dim"] * float(runner.config["partial_rotary_factor"]))
    q = runner._randn(dims["batch"], dims["seq"], dims["q_heads"], rotary_dim, requires_grad=True)
    k = runner._randn(dims["batch"], dims["seq"], dims["kv_heads"], rotary_dim, requires_grad=True)
    freqs = runner.torch.arange(0, rotary_dim, 2, device=runner.device, dtype=runner.torch.float32) / rotary_dim
    pos = runner.torch.arange(dims["seq"], device=runner.device, dtype=runner.torch.float32)
    angles = pos[:, None] * (10000 ** -freqs[None, :])
    cos = angles.cos()[None, :, None, :]
    sin = angles.sin()[None, :, None, :]

    def apply_rope(x):
        even, odd = x[..., 0::2].float(), x[..., 1::2].float()
        return runner.torch.stack(
            (even * cos - odd * sin, odd * cos + even * sin), dim=-1
        ).flatten(-2).to(runner.dtype)

    def fn():
        (apply_rope(q).sum() + apply_rope(k).sum()).float().backward()
        runner._zeros_grad(q, k)

    return runner._measure(plan, fn)


@register_benchmark_case("torch_attention_output_gate")
def run_attention_output_gate(runner, plan):
    dims = _qwen35_dims(runner.config)
    width = dims["q_heads"] * dims["head_dim"]
    x = runner._randn(dims["tokens"], width, requires_grad=True)
    gate = runner._randn(dims["tokens"], width, requires_grad=True)

    def fn():
        (x * runner.torch.sigmoid(gate)).float().sum().backward()
        runner._zeros_grad(x, gate)

    return runner._measure(plan, fn)


@register_benchmark_case("torch_gdn_causal_conv1d")
def run_gdn_causal_conv1d(runner, plan):
    dims = _qwen35_dims(runner.config)
    channels = 2 * dims["key_heads"] * dims["key_head_dim"] + dims["value_heads"] * dims["value_head_dim"]
    kernel = int(runner.config["linear_conv_kernel_dim"])
    module = runner.torch.nn.Conv1d(
        channels,
        channels,
        kernel,
        padding=kernel - 1,
        groups=channels,
        device=runner.device,
        dtype=runner.dtype,
    )
    x = runner._randn(dims["batch"], channels, dims["seq"], requires_grad=True)

    def fn():
        runner.torch.nn.functional.silu(module(x)[..., : dims["seq"]]).float().sum().backward()
        module.zero_grad(set_to_none=True)
        x.grad = None

    return runner._measure(plan, fn)


@register_benchmark_case("torch_gdn_gate_preprocess")
def run_gdn_gate_preprocess(runner, plan):
    dims = _qwen35_dims(runner.config)
    shape = (dims["batch"], dims["seq"], dims["value_heads"])
    a = runner._randn(*shape, requires_grad=True)
    b = runner._randn(*shape, requires_grad=True)

    def fn():
        y = runner.torch.sigmoid(b) - runner.torch.nn.functional.softplus(a.float()).to(runner.dtype)
        y.float().sum().backward()
        runner._zeros_grad(a, b)

    return runner._measure(plan, fn)


@register_benchmark_case("torch_gdn_qk_l2norm")
def run_gdn_qk_l2norm(runner, plan):
    dims = _qwen35_dims(runner.config)
    shape = (dims["batch"], dims["seq"], dims["key_heads"], dims["key_head_dim"])
    q = runner._randn(*shape, requires_grad=True)
    k = runner._randn(*shape, requires_grad=True)

    def fn():
        y = runner.torch.nn.functional.normalize(q.float(), dim=-1)
        y = y + runner.torch.nn.functional.normalize(k.float(), dim=-1)
        y.sum().backward()
        runner._zeros_grad(q, k)

    return runner._measure(plan, fn)


@register_benchmark_case("fla_gated_delta_rule")
def run_gated_delta_rule(runner, plan):
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    except Exception as exc:
        from hcu_train_simulator.benchmarks.operators import BenchmarkResult

        return BenchmarkResult(
            plan.name, plan.group, "skipped", None, None, None, None,
            None, None, plan.shape, f"FLA gated-delta kernel unavailable: {exc}",
        )
    dims = _qwen35_dims(runner.config)
    qkv_prefix = (dims["batch"], dims["seq"], dims["value_heads"])
    q = runner._randn(*qkv_prefix, dims["key_head_dim"], requires_grad=True)
    k = runner._randn(*qkv_prefix, dims["key_head_dim"], requires_grad=True)
    v = runner._randn(*qkv_prefix, dims["value_head_dim"], requires_grad=True)
    g = runner._randn(*qkv_prefix, requires_grad=True)
    beta_logits = runner._randn(*qkv_prefix, requires_grad=True)

    def fn():
        beta = runner.torch.sigmoid(beta_logits)
        y = chunk_gated_delta_rule(
            q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True
        )[0]
        y.float().sum().backward()
        runner._zeros_grad(q, k, v, g, beta_logits)

    return runner._measure(plan, fn)


@register_benchmark_case("torch_gdn_gated_rmsnorm")
def run_gdn_gated_rmsnorm(runner, plan):
    dims = _qwen35_dims(runner.config)
    shape = (
        dims["batch"], dims["seq"], dims["value_heads"], dims["value_head_dim"]
    )
    x = runner._randn(*shape, requires_grad=True)
    z = runner._randn(*shape, requires_grad=True)
    weight = runner.torch.ones(
        dims["value_head_dim"], device=runner.device, dtype=runner.dtype,
        requires_grad=True,
    )

    def fn():
        norm = x * runner.torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + 1e-6
        ).to(runner.dtype)
        y = norm * weight * runner.torch.nn.functional.silu(z)
        y.float().sum().backward()
        runner._zeros_grad(x, z, weight)

    return runner._measure(plan, fn)


def run_qwen35_te_linear(runner, plan):
    import transformer_engine.pytorch as te

    config = runner.config
    dims = _qwen35_dims(config)
    key_dim = dims["key_heads"] * dims["key_head_dim"]
    value_dim = dims["value_heads"] * dims["value_head_dim"]
    shapes = {
        "te_qwen35_q_proj_gate": (dims["hidden"], 2 * dims["q_heads"] * dims["head_dim"]),
        "te_qwen35_k_proj": (dims["hidden"], dims["kv_heads"] * dims["head_dim"]),
        "te_qwen35_v_proj": (dims["hidden"], dims["kv_heads"] * dims["head_dim"]),
        "te_gdn_in_proj_qkv": (dims["hidden"], 2 * key_dim + value_dim),
        "te_gdn_in_proj_z": (dims["hidden"], value_dim),
        "te_gdn_in_proj_a": (dims["hidden"], dims["value_heads"]),
        "te_gdn_in_proj_b": (dims["hidden"], dims["value_heads"]),
        "te_gdn_out_proj": (value_dim, dims["hidden"]),
    }
    in_features, out_features = shapes[plan.name]
    local_tokens = dims["tokens"]
    fused_norm = plan.name in {"te_qwen35_q_proj_gate", "te_gdn_in_proj_qkv"}
    module_type = te.LayerNormLinear if fused_norm else te.Linear
    kwargs = {"bias": False}
    if fused_norm:
        kwargs["normalization"] = "RMSNorm"
    module = runner._make_te_module(module_type, in_features, out_features, **kwargs)
    x = runner._randn(local_tokens, in_features, requires_grad=True)
    return runner._module_bwd(
        plan, module, x,
        flops_per_iter=6 * local_tokens * in_features * out_features,
    )


for _case_name in (
    "te_qwen35_q_proj_gate", "te_qwen35_k_proj", "te_qwen35_v_proj",
    "te_gdn_in_proj_qkv", "te_gdn_in_proj_z", "te_gdn_in_proj_a",
    "te_gdn_in_proj_b", "te_gdn_out_proj",
):
    register_benchmark_case(_case_name)(run_qwen35_te_linear)
