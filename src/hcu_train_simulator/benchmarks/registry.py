# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Benchmark extension registry used by model and operator plugins."""

from __future__ import annotations


_MODULE_MAPPING: dict[str, str | list[str]] = {
    "qkv_weight": "te_linear_qkv",
    "attention_rmsnorm": "te_rmsnorm",
    "qk_norm": "torch_qk_rmsnorm",
    "rope": ["te_fused_rope", "torch_rope"],
    "attention_softmax": "torch_softmax_dropout",
    "flash_attn": "flash_attention",
    "attn_proj": "te_linear_proj",
    "mlp_rmsnorm": "te_rmsnorm",
    "linear_fc1": "te_dense_linear_fc1",
    "swiglu_activation": "torch_swiglu_activation",
    "linear_fc2": "te_dense_linear_fc2",
    "topk_router": "torch_moe_router",
    "router_select": "torch_moe_router",
    "moe_swiglu_activation": "torch_moe_swiglu_activation",
    "shared_expert_fc1": "te_shared_experts_linear_fc1",
    "shared_swiglu_activation": "torch_shared_swiglu_activation",
    "shared_expert_fc2": "te_shared_experts_linear_fc2",
    "shared_expert_gate": "te_shared_expert_gate",
    "mtp_pre_fc_norm_embedding": "te_rmsnorm",
    "mtp_pre_fc_norm_hidden": "te_rmsnorm",
    "mtp_fc": "te_mtp_fc",
    "mtp_final_norm": "te_rmsnorm",
    "lm_head": "torch_lm_head",
    "final_rmsnorm": "te_rmsnorm",
    "cross_entropy": ["torch_cross_entropy", "torch_vocab_parallel_cross_entropy"],
    "optimizer_step": "torch_adamw_step",
}
_MODEL_MODULE_MAPPINGS: dict[str, dict[str, str | list[str]]] = {}
_PRESETS: dict[str, dict] = {}
_PLAN_BUILDERS: list[object] = []
_CASE_RUNNERS: dict[str, object] = {}


def register_benchmark_mapping(
    mapping: dict[str, str | list[str]],
    *,
    model_adapter: str | None = None,
) -> None:
    target = (
        _MODULE_MAPPING
        if model_adapter is None
        else _MODEL_MODULE_MAPPINGS.setdefault(model_adapter, {})
    )
    for model_part, benchmark_case in mapping.items():
        existing = target.get(model_part)
        if existing is not None and existing != benchmark_case:
            raise ValueError(f"benchmark mapping for {model_part!r} is already registered")
        target[model_part] = benchmark_case


def benchmark_module_mapping(model_adapter: str | None = None) -> dict[str, str | list[str]]:
    # Loading model adapters also loads their cost/compute/benchmark extensions.
    from hcu_train_simulator.models.registry import registered_model_adapters

    registered_model_adapters()
    mapping = dict(_MODULE_MAPPING)
    if model_adapter is not None:
        mapping.update(_MODEL_MODULE_MAPPINGS.get(model_adapter, {}))
    return mapping


def register_benchmark_preset(name: str, config: dict) -> None:
    existing = _PRESETS.get(name)
    if existing is not None and existing != config:
        raise ValueError(f"benchmark preset {name!r} is already registered")
    _PRESETS[name] = dict(config)


def benchmark_presets() -> dict[str, dict]:
    benchmark_module_mapping()
    return {name: dict(config) for name, config in _PRESETS.items()}


def register_plan_builder(builder):
    if builder not in _PLAN_BUILDERS:
        _PLAN_BUILDERS.append(builder)
    return builder


def registered_plan_builders():
    benchmark_module_mapping()
    return tuple(_PLAN_BUILDERS)


def register_benchmark_case(name: str):
    def decorator(handler):
        existing = _CASE_RUNNERS.get(name)
        if existing is not None and existing is not handler:
            raise ValueError(f"benchmark runner for {name!r} is already registered")
        _CASE_RUNNERS[name] = handler
        return handler

    return decorator


def get_benchmark_case(name: str):
    benchmark_module_mapping()
    return _CASE_RUNNERS.get(name)
