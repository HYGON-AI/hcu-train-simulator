# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Qwen3.5 compute lowering registrations."""

from hcu_train_simulator.modeling.architecture import query_projection_dim
from hcu_train_simulator.modeling.compute_registry import (
    register_attention_lowering,
    register_component_lowering,
    register_gemm_shape,
)
from hcu_train_simulator.modeling.spec import (
    GatedDeltaNet,
    GatedSelfAttention,
    MultiTokenPredictor,
)
from hcu_train_simulator.parallelism import get_num_microbatches


def _gdn_key_dim(model):
    return model.linear_num_key_heads * model.linear_key_head_dim


def _gdn_value_dim(model):
    return model.linear_num_value_heads * model.linear_value_head_dim


def _microbatch_compute_count(model):
    return get_num_microbatches(
        model.global_batch_size,
        model.micro_batch_size,
        model.dp_size,
    )


@register_gemm_shape("qwen35_q_proj_gate")
def qwen35_q_proj_gate_shape(model):
    return (
        model.transformer_compute_count(),
        2 * query_projection_dim(model) // model.tp_size,
        model.hidden_size,
        model.token_count(),
    )


def _qwen35_kv_shape(model):
    return (
        model.transformer_compute_count(),
        model.num_query_groups * model.kv_channels // model.tp_size,
        model.hidden_size,
        model.token_count(),
    )


register_gemm_shape("qwen35_k_proj")(_qwen35_kv_shape)
register_gemm_shape("qwen35_v_proj")(_qwen35_kv_shape)


@register_gemm_shape("gdn_in_proj_qkv")
def gdn_in_proj_qkv_shape(model):
    out = (2 * _gdn_key_dim(model) + _gdn_value_dim(model)) // model.tp_size
    return model.transformer_compute_count(), out, model.hidden_size, model.token_count()


@register_gemm_shape("gdn_in_proj_z")
def gdn_in_proj_z_shape(model):
    return (
        model.transformer_compute_count(),
        _gdn_value_dim(model) // model.tp_size,
        model.hidden_size,
        model.token_count(),
    )


def _gdn_ab_shape(model):
    return (
        model.transformer_compute_count(),
        model.linear_num_value_heads // model.tp_size,
        model.hidden_size,
        model.token_count(),
    )


register_gemm_shape("gdn_in_proj_a")(_gdn_ab_shape)
register_gemm_shape("gdn_in_proj_b")(_gdn_ab_shape)


@register_gemm_shape("gdn_out_proj")
def gdn_out_proj_shape(model):
    return (
        model.transformer_compute_count(),
        model.hidden_size,
        _gdn_value_dim(model) // model.tp_size,
        model.token_count(),
    )


@register_gemm_shape("mtp_fc")
def mtp_fc_shape(model):
    return (
        _microbatch_compute_count(model),
        model.hidden_size // model.tp_size,
        2 * model.hidden_size,
        model.token_count(),
    )


@register_attention_lowering(GatedSelfAttention, first_gemm_part="qwen35_q_proj_gate")
def lower_gated_attention(model, result, attention, module_path, compute_count, count_scale):
    gated = attention.submodules
    for part, part_spec, part_name in (
        ("qwen35_q_proj_gate", gated.linear_q, "linear_q"),
        ("qwen35_k_proj", gated.linear_k, "linear_k"),
        ("qwen35_v_proj", gated.linear_v, "linear_v"),
    ):
        model.add_gemm_part(
            result,
            part,
            count_scale=count_scale,
            module_spec=part_spec,
            module_path=f"{module_path}.self_attention.{part_name}",
        )
    model.add_flash_attention_after(
        result,
        "qkv_weight",
        count_scale=count_scale,
        mla=False,
        module_spec=gated.core_attention,
        module_path=f"{module_path}.self_attention.core_attention",
    )
    model.add_common_attention_non_gemm(
        result,
        compute_count,
        attention_spec=attention,
        module_path=f"{module_path}.self_attention",
    )
    model.add_non_gemm_part(
        result,
        "attention_output_gate",
        model.transformer_element_count(query_projection_dim(model), tp_sharded=True),
        compute_count,
        "sigmoid_gate",
        module_spec=gated.output_gate,
        module_path=f"{module_path}.self_attention.output_gate",
    )
    model.add_gemm_part(
        result,
        "attn_proj",
        count_scale=count_scale,
        module_spec=gated.linear_proj,
        module_path=f"{module_path}.self_attention.linear_proj",
    )


@register_attention_lowering(GatedDeltaNet, first_gemm_part="gdn_in_proj_qkv")
def lower_gated_delta_net(model, result, attention, module_path, compute_count, count_scale):
    gdn = attention.submodules
    for part, part_spec, part_name in (
        ("gdn_in_proj_qkv", gdn.in_proj_qkv, "in_proj_qkv"),
        ("gdn_in_proj_z", gdn.in_proj_z, "in_proj_z"),
        ("gdn_in_proj_b", gdn.in_proj_b, "in_proj_b"),
        ("gdn_in_proj_a", gdn.in_proj_a, "in_proj_a"),
    ):
        model.add_gemm_part(
            result,
            part,
            count_scale=count_scale,
            module_spec=part_spec,
            module_path=f"{module_path}.self_attention.{part_name}",
        )
    key_dim = _gdn_key_dim(model)
    value_dim = _gdn_value_dim(model)
    model.add_non_gemm_part(
        result,
        "gdn_causal_conv1d",
        model.token_count() * (2 * key_dim + value_dim) / model.tp_size,
        compute_count,
        "causal_conv1d",
        module_spec=gdn.conv1d,
        module_path=f"{module_path}.self_attention.conv1d",
    )
    model.add_non_gemm_part(
        result,
        "gdn_gate_preprocess",
        model.token_count() * 2 * model.linear_num_value_heads / model.tp_size,
        compute_count,
        "sigmoid_gate",
        module_spec=attention,
        module_path=f"{module_path}.self_attention.gates",
    )
    model.add_non_gemm_part(
        result,
        "gdn_qk_l2norm",
        model.token_count() * 2 * key_dim / model.tp_size,
        compute_count,
        "l2norm",
        module_spec=gdn.core,
        module_path=f"{module_path}.self_attention.core.qk_l2norm",
    )
    model.add_non_gemm_part(
        result,
        "gated_delta_rule",
        model.token_count() * model.linear_num_value_heads * model.linear_key_head_dim * model.linear_value_head_dim / model.tp_size,
        compute_count,
        "gated_delta_rule",
        module_spec=gdn.core,
        module_path=f"{module_path}.self_attention.core",
    )
    model.add_non_gemm_part(
        result,
        "gdn_gated_rmsnorm",
        model.token_count() * value_dim / model.tp_size,
        compute_count,
        "gated_rmsnorm",
        module_spec=gdn.gated_norm,
        module_path=f"{module_path}.self_attention.gated_norm",
    )
    model.add_gemm_part(
        result,
        "gdn_out_proj",
        count_scale=count_scale,
        module_spec=gdn.out_proj,
        module_path=f"{module_path}.self_attention.out_proj",
    )


@register_component_lowering(MultiTokenPredictor)
def lower_mtp(model, result, mtp_spec, module_path):
    submodules = mtp_spec.submodules
    compute_count = _microbatch_compute_count(model)
    elements = model.token_count() * model.hidden_size
    for part, spec, name in (
        (
            "mtp_pre_fc_norm_embedding",
            submodules.pre_fc_norm_embedding,
            "pre_fc_norm_embedding",
        ),
        (
            "mtp_pre_fc_norm_hidden",
            submodules.pre_fc_norm_hidden,
            "pre_fc_norm_hidden",
        ),
    ):
        model.add_non_gemm_part(
            result,
            part,
            elements,
            compute_count,
            "rmsnorm",
            module_spec=spec,
            module_path=f"{module_path}.{name}",
        )
    model.add_gemm_part(
        result,
        "mtp_fc",
        module_spec=submodules.linear_fc,
        module_path=f"{module_path}.linear_fc",
    )
    model.add_non_gemm_part(
        result,
        "mtp_final_norm",
        elements,
        compute_count,
        "rmsnorm",
        module_spec=submodules.norm,
        module_path=f"{module_path}.norm",
    )
