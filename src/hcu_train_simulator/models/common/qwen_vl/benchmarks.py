# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Measured-kernel support shared by Qwen visual encoders."""

from hcu_train_simulator.benchmarks.registry import (
    register_benchmark_case,
    register_benchmark_mapping,
    register_plan_builder,
)


VISION_BENCHMARK_MAPPING = {
    "vision_patch_embed": "torch_vision_patch_embed",
    "vision_position_add": "torch_vision_position_add",
    "vision_norm1": "torch_vision_layernorm",
    "vision_norm2": "torch_vision_layernorm",
    "vision_qkv": "te_vision_qkv",
    "vision_rope": "torch_vision_rope",
    "vision_flash_attn": "flash_vision_attention",
    "vision_attn_proj": "te_vision_attn_proj",
    "vision_fc1": "te_vision_fc1",
    "vision_gelu": "torch_vision_gelu",
    "vision_fc2": "te_vision_fc2",
    "vision_residual_add": "torch_vision_residual_add",
    "vision_merger_norm": "torch_vision_layernorm",
    "vision_merger_fc1": "te_vision_merger_fc1",
    "vision_merger_gelu": "torch_vision_merger_gelu",
    "vision_merger_fc2": "te_vision_merger_fc2",
    "vision_deepstack_merger_norm": "torch_vision_deepstack_layernorm",
    "vision_deepstack_merger_fc1": "te_vision_merger_fc1",
    "vision_deepstack_merger_gelu": "torch_vision_merger_gelu",
    "vision_deepstack_merger_fc2": "te_vision_merger_fc2",
    "vision_deepstack_add": "torch_vision_residual_add",
}


def register_vision_benchmark_mapping(model_adapter):
    register_benchmark_mapping(
        VISION_BENCHMARK_MAPPING,
        model_adapter=model_adapter,
    )


@register_plan_builder
def build_qwen_vision_operator_plans(config, plan_type):
    if not config.get("has_vision"):
        return []

    tp = int(config["tp_size"])
    batch = int(config["micro_batch_size"])
    vision_hidden = int(config["vision_hidden_size"])
    vision_ffn = int(config["vision_ffn_hidden_size"])
    vision_heads = int(config["vision_num_heads"])
    vision_seq = int(config["vision_seq_length"])
    vision_batch = batch * int(config.get("vision_num_images", 1))
    vision_tokens = vision_batch * vision_seq
    merge = int(config["vision_spatial_merge_size"])
    merged_hidden = vision_hidden * merge**2
    plans = [
        plan_type(
            "torch_vision_patch_embed",
            "vision",
            "Qwen Conv3d patch embedding",
            (
                f"patches={vision_tokens}, "
                f"channels={config['vision_in_channels']}, "
                f"kernel={config['vision_temporal_patch_size']}x"
                f"{config['vision_patch_size']}x"
                f"{config['vision_patch_size']}"
            ),
        ),
        plan_type(
            "torch_vision_position_add",
            "vision",
            "Vision learned position embedding add forward+backward",
            f"[{vision_batch},{vision_seq},{vision_hidden}]",
        ),
        plan_type(
            "te_vision_qkv",
            "vision",
            "Vision QKV projection",
            (
                f"[{vision_tokens},{vision_hidden}] x "
                f"[{vision_hidden},{3 * vision_hidden // tp}]"
            ),
            "transformer_engine",
        ),
        plan_type(
            "torch_vision_rope",
            "vision",
            "Vision rotary embedding forward+backward",
            (
                f"q/k=[{vision_batch},{vision_seq},"
                f"{vision_heads // tp},"
                f"{vision_hidden // vision_heads}]"
            ),
        ),
        plan_type(
            "flash_vision_attention",
            "vision",
            "Non-causal vision Flash Attention forward+backward",
            (
                f"qkv=[{vision_batch},{vision_seq},"
                f"{vision_heads // tp},"
                f"{vision_hidden // vision_heads}]"
            ),
            "flash_attn",
        ),
        plan_type(
            "te_vision_attn_proj",
            "vision",
            "Vision attention output projection",
            (
                f"[{vision_tokens},{vision_hidden // tp}] x "
                f"[{vision_hidden // tp},{vision_hidden}]"
            ),
            "transformer_engine",
        ),
        plan_type(
            "te_vision_fc1",
            "vision",
            "Vision GELU MLP FC1",
            (
                f"[{vision_tokens},{vision_hidden}] x "
                f"[{vision_hidden},{vision_ffn // tp}]"
            ),
            "transformer_engine",
        ),
        plan_type(
            "torch_vision_gelu",
            "vision",
            "Vision GELU forward+backward",
            f"[{vision_tokens},{vision_ffn // tp}]",
        ),
        plan_type(
            "te_vision_fc2",
            "vision",
            "Vision GELU MLP FC2",
            (
                f"[{vision_tokens},{vision_ffn // tp}] x "
                f"[{vision_ffn // tp},{vision_hidden}]"
            ),
            "transformer_engine",
        ),
        plan_type(
            "torch_vision_residual_add",
            "vision",
            "Vision residual add forward+backward",
            f"[{vision_batch},{vision_seq},{vision_hidden}]",
        ),
        plan_type(
            "torch_vision_layernorm",
            "vision",
            "Vision LayerNorm forward+backward",
            f"[{vision_batch},{vision_seq},{vision_hidden}]",
        ),
        plan_type(
            "te_vision_merger_fc1",
            "vision",
            "Vision patch merger FC1",
            (
                f"[{vision_tokens // merge**2},{merged_hidden}] x "
                f"[{merged_hidden},{merged_hidden // tp}]"
            ),
            "transformer_engine",
        ),
        plan_type(
            "torch_vision_merger_gelu",
            "vision",
            "Vision patch merger GELU forward+backward",
            (
                f"[{vision_tokens // merge**2},"
                f"{merged_hidden // tp}]"
            ),
        ),
        plan_type(
            "te_vision_merger_fc2",
            "vision",
            "Vision patch merger FC2",
            (
                f"[{vision_tokens // merge**2},"
                f"{merged_hidden // tp}] x "
                f"[{merged_hidden // tp},"
                f"{config['vision_output_hidden_size']}]"
            ),
            "transformer_engine",
        ),
    ]
    if int(config.get("vision_deepstack_count", 0)):
        plans.append(
            plan_type(
                "torch_vision_deepstack_layernorm",
                "vision",
                "Vision DeepStack post-shuffle LayerNorm forward+backward",
                f"[{vision_tokens // merge**2},{merged_hidden}]",
            )
        )
    return plans


def _vision_dims(config):
    batch = (
        int(config["micro_batch_size"])
        * int(config.get("vision_num_images", 1))
    )
    seq = int(config["vision_seq_length"])
    hidden = int(config["vision_hidden_size"])
    heads = int(config["vision_num_heads"]) // int(config["tp_size"])
    return batch, seq, hidden, heads


@register_benchmark_case("torch_vision_patch_embed")
def run_vision_patch_embed(runner, plan):
    batch, seq, hidden, _ = _vision_dims(runner.config)
    channels = int(runner.config["vision_in_channels"])
    temporal = int(runner.config["vision_temporal_patch_size"])
    patch = int(runner.config["vision_patch_size"])
    module = runner.torch.nn.Conv3d(
        channels,
        hidden,
        (temporal, patch, patch),
        stride=(temporal, patch, patch),
        device=runner.device,
        dtype=runner.dtype,
    )
    x = runner._randn(
        batch * seq,
        channels,
        temporal,
        patch,
        patch,
        requires_grad=True,
    )
    return runner._module_bwd(plan, module, x)


def run_vision_add(runner, plan):
    batch, seq, hidden, _ = _vision_dims(runner.config)
    x = runner._randn(batch, seq, hidden, requires_grad=True)
    residual = runner._randn(
        batch,
        seq,
        hidden,
        requires_grad=True,
    )

    def fn():
        (x + residual).float().sum().backward()
        runner._zeros_grad(x, residual)

    return runner._measure(plan, fn)


register_benchmark_case("torch_vision_position_add")(run_vision_add)
register_benchmark_case("torch_vision_residual_add")(run_vision_add)


@register_benchmark_case("torch_vision_layernorm")
def run_vision_layernorm(runner, plan):
    batch, seq, hidden, _ = _vision_dims(runner.config)
    module = runner.torch.nn.LayerNorm(
        hidden,
        device=runner.device,
        dtype=runner.dtype,
    )
    x = runner._randn(batch, seq, hidden, requires_grad=True)
    return runner._module_bwd(plan, module, x)


@register_benchmark_case("torch_vision_deepstack_layernorm")
def run_vision_deepstack_layernorm(runner, plan):
    config = runner.config
    batch, seq, hidden, _ = _vision_dims(config)
    merge = int(config["vision_spatial_merge_size"])
    merged_hidden = hidden * merge**2
    module = runner.torch.nn.LayerNorm(
        merged_hidden,
        device=runner.device,
        dtype=runner.dtype,
    )
    x = runner._randn(
        batch * seq // merge**2,
        merged_hidden,
        requires_grad=True,
    )
    return runner._module_bwd(plan, module, x)


def run_vision_gelu(runner, plan):
    config = runner.config
    batch, seq, hidden, _ = _vision_dims(config)
    width = (
        int(config["vision_ffn_hidden_size"])
        // int(config["tp_size"])
    )
    count = batch * seq
    if plan.name == "torch_vision_merger_gelu":
        merge = int(config["vision_spatial_merge_size"])
        width = hidden * merge**2 // int(config["tp_size"])
        count //= merge**2
    x = runner._randn(count, width, requires_grad=True)

    def fn():
        runner.torch.nn.functional.gelu(
            x,
            approximate="tanh",
        ).float().sum().backward()
        x.grad = None

    return runner._measure(plan, fn)


register_benchmark_case("torch_vision_gelu")(run_vision_gelu)
register_benchmark_case("torch_vision_merger_gelu")(run_vision_gelu)


def run_vision_attention_aux(runner, plan):
    batch, seq, hidden, heads = _vision_dims(runner.config)
    dim = hidden // int(runner.config["vision_num_heads"])
    q = runner._randn(
        batch,
        seq,
        heads,
        dim,
        requires_grad=True,
    )
    k = runner._randn(
        batch,
        seq,
        heads,
        dim,
        requires_grad=True,
    )
    if plan.name == "flash_vision_attention":
        flash_attn_func = runner._import_flash_attention()
        v = runner._randn(
            batch,
            seq,
            heads,
            dim,
            requires_grad=True,
        )

        def fn():
            flash_attn_func(
                q,
                k,
                v,
                dropout_p=0.0,
                causal=False,
            ).float().sum().backward()
            runner._zeros_grad(q, k, v)

        return runner._measure(
            plan,
            fn,
            flops_per_iter=4 * batch * heads * seq * seq * dim,
        )

    freqs = (
        runner.torch.arange(
            0,
            dim,
            2,
            device=runner.device,
            dtype=runner.torch.float32,
        )
        / dim
    )
    pos = runner.torch.arange(
        seq,
        device=runner.device,
        dtype=runner.torch.float32,
    )
    angles = pos[:, None] * (10000 ** -freqs[None, :])
    cos = angles.cos()[None, :, None, :]
    sin = angles.sin()[None, :, None, :]

    def apply_rope(x):
        even = x[..., 0::2].float()
        odd = x[..., 1::2].float()
        return runner.torch.stack(
            (even * cos - odd * sin, odd * cos + even * sin),
            -1,
        ).flatten(-2).to(runner.dtype)

    def fn():
        (apply_rope(q).sum() + apply_rope(k).sum()).float().backward()
        runner._zeros_grad(q, k)

    return runner._measure(plan, fn)


register_benchmark_case("torch_vision_rope")(run_vision_attention_aux)
register_benchmark_case("flash_vision_attention")(
    run_vision_attention_aux
)


def run_vision_te_linear(runner, plan):
    import transformer_engine.pytorch as te

    config = runner.config
    tp = int(config["tp_size"])
    vision_hidden = int(config["vision_hidden_size"])
    vision_ffn = int(config["vision_ffn_hidden_size"])
    merge = int(config["vision_spatial_merge_size"])
    merged_hidden = vision_hidden * merge**2
    shapes = {
        "te_vision_qkv": (
            vision_hidden,
            3 * vision_hidden // tp,
        ),
        "te_vision_attn_proj": (
            vision_hidden // tp,
            vision_hidden,
        ),
        "te_vision_fc1": (
            vision_hidden,
            vision_ffn // tp,
        ),
        "te_vision_fc2": (
            vision_ffn // tp,
            vision_hidden,
        ),
        "te_vision_merger_fc1": (
            merged_hidden,
            merged_hidden // tp,
        ),
        "te_vision_merger_fc2": (
            merged_hidden // tp,
            int(config["vision_output_hidden_size"]),
        ),
    }
    in_features, out_features = shapes[plan.name]
    local_tokens = (
        int(config["micro_batch_size"])
        * int(config.get("vision_num_images", 1))
        * int(config["vision_seq_length"])
    )
    if "merger" in plan.name:
        local_tokens //= merge**2
    module = runner._make_te_module(
        te.Linear,
        in_features,
        out_features,
        bias=False,
    )
    x = runner._randn(
        local_tokens,
        in_features,
        requires_grad=True,
    )
    return runner._module_bwd(
        plan,
        module,
        x,
        flops_per_iter=(
            6 * local_tokens * in_features * out_features
        ),
    )


for _case_name in (
    "te_vision_qkv",
    "te_vision_attn_proj",
    "te_vision_fc1",
    "te_vision_fc2",
    "te_vision_merger_fc1",
    "te_vision_merger_fc2",
):
    register_benchmark_case(_case_name)(run_vision_te_linear)
