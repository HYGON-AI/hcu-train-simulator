# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""GLM-5 DSA operator benchmark plans and runnable cases."""

from hcu_train_simulator.benchmarks.registry import (
    register_benchmark_case,
    register_benchmark_mapping,
    register_plan_builder,
)


register_benchmark_mapping(
    {
        "dsa_index_q": "te_glm5_dsa_index_q",
        "dsa_index_k": "te_glm5_dsa_index_k",
        "dsa_index_weights": "te_glm5_dsa_index_weights",
        "dsa_index_score": "torch_glm5_dsa_index_score",
        "dsa_index_k_norm": "torch_glm5_dsa_index_k_norm",
        "dsa_topk": "torch_glm5_dsa_topk",
        "dsa_index_rope": "torch_glm5_dsa_index_rope",
    },
    model_adapter="glm5",
)


def _dims(config):
    return {
        "batch": int(config["micro_batch_size"]),
        "seq": int(config["local_seq_length"]),
        "kv_seq": int(config["attention_kv_seq_length"]),
        "tokens": int(config["tokens"]),
        "hidden": int(config["hidden_size"]),
        "q_lora_rank": int(config["q_lora_rank"]),
        "heads": int(config["index_n_heads"]),
        "head_dim": int(config["index_head_dim"]),
        "rope_dim": int(config["qk_rope_head_dim"]),
        "topk": min(int(config["index_topk"]), int(config["attention_kv_seq_length"])),
    }


@register_plan_builder
def build_glm5_dsa_operator_plans(config, plan_type):
    if not config.get("has_dsa"):
        return []
    dims = _dims(config)
    return [
        plan_type(
            "te_glm5_dsa_index_q",
            "dsa",
            "GLM-5 replicated DSA index query projection",
            f"[{dims['tokens']},{dims['q_lora_rank']}] x "
            f"[{dims['q_lora_rank']},{dims['heads'] * dims['head_dim']}]",
            "transformer_engine",
        ),
        plan_type(
            "te_glm5_dsa_index_k",
            "dsa",
            "GLM-5 replicated DSA index key projection",
            f"[{dims['tokens']},{dims['hidden']}] x "
            f"[{dims['hidden']},{dims['head_dim']}]",
            "transformer_engine",
        ),
        plan_type(
            "te_glm5_dsa_index_weights",
            "dsa",
            "GLM-5 DSA per-head weight projection",
            f"[{dims['tokens']},{dims['hidden']}] x "
            f"[{dims['hidden']},{dims['heads']}]",
            "transformer_engine",
        ),
        plan_type(
            "torch_glm5_dsa_index_k_norm",
            "dsa",
            "GLM-5 DSA index key LayerNorm",
            f"[{dims['batch']},{dims['seq']},{dims['head_dim']}]",
        ),
        plan_type(
            "torch_glm5_dsa_index_rope",
            "dsa",
            "GLM-5 interleaved indexer RoPE",
            f"q=[{dims['batch']},{dims['seq']},{dims['heads']},{dims['rope_dim']}], "
            f"k=[{dims['batch']},{dims['seq']},1,{dims['rope_dim']}]",
        ),
        plan_type(
            "torch_glm5_dsa_index_score",
            "dsa",
            "GLM-5 DSA QK score, ReLU, and weighted head reduction",
            f"q=[{dims['batch']},{dims['seq']},{dims['heads']},{dims['head_dim']}], "
            f"k=[{dims['batch']},{dims['kv_seq']},{dims['head_dim']}]",
        ),
        plan_type(
            "torch_glm5_dsa_topk",
            "dsa",
            "GLM-5 DSA token top-k selection",
            f"scores=[{dims['batch']},{dims['seq']},{dims['kv_seq']}], topk={dims['topk']}",
        ),
    ]


def run_dsa_linear(runner, plan):
    import transformer_engine.pytorch as te

    dims = _dims(runner.config)
    shapes = {
        "te_glm5_dsa_index_q": (
            dims["q_lora_rank"],
            dims["heads"] * dims["head_dim"],
        ),
        "te_glm5_dsa_index_k": (dims["hidden"], dims["head_dim"]),
        "te_glm5_dsa_index_weights": (dims["hidden"], dims["heads"]),
    }
    in_features, out_features = shapes[plan.name]
    module = runner._make_te_module(te.Linear, in_features, out_features, bias=False)
    x = runner._randn(dims["tokens"], in_features, requires_grad=True)
    return runner._module_bwd(
        plan,
        module,
        x,
        flops_per_iter=6 * dims["tokens"] * in_features * out_features,
    )


for _case_name in (
    "te_glm5_dsa_index_q",
    "te_glm5_dsa_index_k",
    "te_glm5_dsa_index_weights",
):
    register_benchmark_case(_case_name)(run_dsa_linear)


@register_benchmark_case("torch_glm5_dsa_index_k_norm")
def run_dsa_index_k_norm(runner, plan):
    dims = _dims(runner.config)
    module = runner.torch.nn.LayerNorm(
        dims["head_dim"],
        device=runner.device,
        dtype=runner.dtype,
    )
    x = runner._randn(
        dims["batch"], dims["seq"], dims["head_dim"], requires_grad=True
    )
    return runner._module_bwd(plan, module, x)


@register_benchmark_case("torch_glm5_dsa_index_score")
def run_dsa_index_score(runner, plan):
    dims = _dims(runner.config)
    q = runner._randn(
        dims["batch"],
        dims["seq"],
        dims["heads"],
        dims["head_dim"],
        requires_grad=True,
    )
    k = runner._randn(
        dims["batch"], dims["kv_seq"], dims["head_dim"], requires_grad=True
    )
    weights = runner._randn(
        dims["batch"], dims["seq"], dims["heads"], requires_grad=True
    )

    def fn():
        scores = runner.torch.einsum("bshd,btd->bsht", q, k).float().relu()
        reduced = runner.torch.einsum("bsh,bsht->bst", weights.float(), scores)
        reduced.sum().backward()
        runner._zeros_grad(q, k, weights)

    flops = (
        6
        * dims["batch"]
        * dims["seq"]
        * dims["kv_seq"]
        * dims["heads"]
        * dims["head_dim"]
    )
    return runner._measure(plan, fn, flops_per_iter=flops)


@register_benchmark_case("torch_glm5_dsa_topk")
def run_dsa_topk(runner, plan):
    dims = _dims(runner.config)
    scores = runner._randn(
        dims["batch"], dims["seq"], dims["kv_seq"], requires_grad=True
    )

    def fn():
        values = scores.topk(dims["topk"], dim=-1, sorted=False).values
        values.float().sum().backward()
        scores.grad = None

    return runner._measure(plan, fn)


@register_benchmark_case("torch_glm5_dsa_index_rope")
def run_dsa_index_rope(runner, plan):
    dims = _dims(runner.config)
    q = runner._randn(
        dims["batch"],
        dims["seq"],
        dims["heads"],
        dims["rope_dim"],
        requires_grad=True,
    )
    k = runner._randn(
        dims["batch"], dims["seq"], 1, dims["rope_dim"], requires_grad=True
    )
    half = dims["rope_dim"] // 2
    angles = runner.torch.arange(
        dims["seq"], device=runner.device, dtype=runner.torch.float32
    )[:, None] * runner.torch.arange(
        half, device=runner.device, dtype=runner.torch.float32
    )[None, :]
    cos = angles.cos()[None, :, None, :]
    sin = angles.sin()[None, :, None, :]

    def apply(x):
        even = x[..., 0::2].float()
        odd = x[..., 1::2].float()
        return runner.torch.stack(
            (even * cos - odd * sin, odd * cos + even * sin), dim=-1
        ).flatten(-2)

    def fn():
        (apply(q).sum() + apply(k).sum()).backward()
        runner._zeros_grad(q, k)

    return runner._measure(plan, fn)
