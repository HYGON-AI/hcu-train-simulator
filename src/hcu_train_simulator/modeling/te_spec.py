# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Default Transformer Engine module-spec builders.

The layout follows Megatron-Core's GPT layer specs, but contains only semantic
module markers and therefore has no runtime dependency on Megatron or TE.
"""

from hcu_train_simulator.modeling.spec import (
    BiasDropoutAdd,
    Embedding,
    GPTModel,
    GPTModelSubmodules,
    IdentityOp,
    MLP,
    MLPSubmodules,
    MLASelfAttention,
    MLASelfAttentionSubmodules,
    MoELayer,
    MoESubmodules,
    MultiTokenPredictor,
    MultiTokenPredictorSubmodules,
    ModuleSpec,
    SelfAttention,
    SelfAttentionSubmodules,
    SharedExpertMLP,
    SwiGLU,
    TEColumnParallelLinear,
    TEDotProductAttention,
    TEGroupedMLP,
    TELayerNormColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
    TESequentialMLP,
    TopKRouter,
    TransformerBlock,
    TransformerBlockSubmodules,
    TransformerLayer,
    TransformerLayerSubmodules,
)
from hcu_train_simulator.modeling.architecture import layer_uses_moe


def _spec(module, *, role=None, **params):
    metainfo = {"role": role} if role else {}
    return ModuleSpec(module=module, params=params, metainfo=metainfo)


def _te_norm(scope="hidden_size", role=None):
    return _spec(TENorm, role=role, normalization="RMSNorm", scope=scope)


def _te_layernorm(scope="hidden_size", role=None):
    return _spec(TENorm, role=role, normalization="LayerNorm", scope=scope)


def _get_self_attention_spec(transformer_config, qk_layernorm):
    q_norm = _te_norm("head_dim", role="q_layernorm") if qk_layernorm else _spec(IdentityOp)
    k_norm = _te_norm("head_dim", role="k_layernorm") if qk_layernorm else _spec(IdentityOp)
    return ModuleSpec(
        module=SelfAttention,
        params={"attn_mask_type": "causal"},
        submodules=SelfAttentionSubmodules(
            linear_qkv=_spec(
                TELayerNormColumnParallelLinear,
                role="qkv_weight",
                fused_norm_scope="hidden_size",
            ),
            core_attention=_spec(TEDotProductAttention, role="flash_attn"),
            linear_proj=_spec(TERowParallelLinear, role="attn_proj"),
            q_layernorm=q_norm,
            k_layernorm=k_norm,
        ),
        metainfo={"backend": "transformer_engine"},
    )


def _get_mla_attention_spec(transformer_config):
    q_lora_rank = transformer_config.q_lora_rank
    return ModuleSpec(
        module=MLASelfAttention,
        params={"attn_mask_type": "causal"},
        submodules=MLASelfAttentionSubmodules(
            linear_q_proj=(
                _spec(TEColumnParallelLinear, role="mla_q_proj", fused_norm_scope="hidden_size")
                if not q_lora_rank else None
            ),
            linear_q_down_proj=(
                _spec(
                    TELayerNormColumnParallelLinear,
                    role="mla_q_down",
                    fused_norm_scope="hidden_size",
                )
                if q_lora_rank else None
            ),
            linear_q_up_proj=_spec(TEColumnParallelLinear, role="mla_q_up"),
            linear_kv_down_proj=_spec(TELayerNormColumnParallelLinear, role="mla_kv_down"),
            linear_kv_up_proj=_spec(TEColumnParallelLinear, role="mla_kv_up"),
            core_attention=_spec(TEDotProductAttention, role="flash_attn"),
            linear_proj=_spec(TERowParallelLinear, role="attn_proj"),
            q_layernorm=_te_norm("q_lora_rank", role="mla_q_layernorm") if q_lora_rank else None,
            kv_layernorm=_te_norm("kv_lora_rank", role="mla_kv_layernorm"),
        ),
        metainfo={"backend": "transformer_engine"},
    )


def _get_dense_mlp_spec():
    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=_spec(
                TELayerNormColumnParallelLinear,
                role="linear_fc1",
                fused_norm_scope="hidden_size",
            ),
            activation_func=_spec(SwiGLU, role="swiglu_activation"),
            linear_fc2=_spec(TERowParallelLinear, role="linear_fc2"),
        ),
        metainfo={"backend": "transformer_engine"},
    )


def _get_moe_spec(transformer_config, moe_grouped_gemm):
    experts = TEGroupedMLP if moe_grouped_gemm else TESequentialMLP
    shared_experts = None
    if transformer_config.num_shared_experts:
        shared_experts = _spec(
            SharedExpertMLP,
            role="shared_experts",
            num_shared_experts=transformer_config.num_shared_experts,
            gated=transformer_config.moe_shared_expert_gate,
        )
    return ModuleSpec(
        module=MoELayer,
        submodules=MoESubmodules(
            router=_spec(TopKRouter, role="topk_router", topk=transformer_config.moe_router_topk),
            experts=_spec(experts, role="routed_experts"),
            shared_experts=shared_experts,
        ),
        metainfo={
            "backend": "transformer_engine",
            "grouped_gemm": bool(moe_grouped_gemm),
        },
    )


def build_multi_token_predictor_spec():
    return ModuleSpec(
        module=MultiTokenPredictor,
        submodules=MultiTokenPredictorSubmodules(
            embedding=_spec(Embedding, role="mtp_embedding"),
            pre_fc_norm_embedding=_te_norm(
                "hidden_size", role="mtp_pre_fc_norm_embedding"
            ),
            pre_fc_norm_hidden=_te_norm(
                "hidden_size", role="mtp_pre_fc_norm_hidden"
            ),
            linear_fc=_spec(TEColumnParallelLinear, role="mtp_fc"),
            norm=_te_norm("hidden_size", role="mtp_final_norm"),
        ),
        metainfo={
            "backend": "transformer_engine",
            "excluded_from_model_size": True,
        },
    )


def get_gpt_layer_with_transformer_engine_spec(
    transformer_config,
    *,
    qk_layernorm=False,
    multi_latent_attention=False,
    num_experts=None,
    moe_grouped_gemm=True,
    is_mtp_layer=False,
    attention_spec=None,
):
    """Build a Megatron-style TransformerLayer ModuleSpec for the TE backend."""

    if attention_spec is not None:
        attention = attention_spec
        attention_variant = attention.metainfo.get("attention_variant")
    elif multi_latent_attention:
        attention = _get_mla_attention_spec(transformer_config)
        attention_variant = "mla"
    else:
        attention = _get_self_attention_spec(transformer_config, qk_layernorm)
        attention_variant = "softmax"

    mlp = (
        _get_moe_spec(transformer_config, moe_grouped_gemm)
        if num_experts
        else _get_dense_mlp_spec()
    )
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            # TE fuses the input norms into the following column-parallel linear.
            input_layernorm=_spec(IdentityOp, role="attention_rmsnorm"),
            self_attention=attention,
            self_attn_bda=_spec(BiasDropoutAdd, role="residual_dropout_add"),
            pre_mlp_layernorm=(
                _te_norm(role="mlp_rmsnorm")
                if num_experts else _spec(IdentityOp, role="mlp_rmsnorm")
            ),
            mlp=mlp,
            mlp_bda=_spec(BiasDropoutAdd, role="residual_dropout_add"),
        ),
        metainfo={
            "backend": "transformer_engine",
            "is_mtp_layer": bool(is_mtp_layer),
            "attention_variant": attention_variant,
        },
    )


def build_transformer_engine_model_spec(
    transformer_config,
    model_config,
    *,
    qk_layernorm=False,
    multi_latent_attention=False,
    layer_attention_specs=None,
    mtp_attention_spec=None,
    mtp_model_spec=None,
    vision_model_spec=None,
    moe_grouped_gemm=True,
):
    """Compose a GPT ModuleSpec from normalized configuration and options."""

    attention_specs = tuple(layer_attention_specs or ())
    if attention_specs and len(attention_specs) != transformer_config.num_layers:
        raise ValueError("layer_attention_specs must contain one entry per transformer layer")

    layer_specs = []
    for index in range(transformer_config.num_layers):
        attention_spec = attention_specs[index] if attention_specs else None
        layer_specs.append(
            get_gpt_layer_with_transformer_engine_spec(
                transformer_config,
                qk_layernorm=qk_layernorm,
                multi_latent_attention=multi_latent_attention,
                num_experts=(
                    transformer_config.num_moe_experts
                    if layer_uses_moe(transformer_config, index)
                    else None
                ),
                moe_grouped_gemm=moe_grouped_gemm,
                attention_spec=attention_spec,
            )
        )
    for mtp_index in range(transformer_config.mtp_num_layers):
        layer_index = transformer_config.num_layers + mtp_index
        layer_specs.append(
            get_gpt_layer_with_transformer_engine_spec(
                transformer_config,
                qk_layernorm=qk_layernorm,
                multi_latent_attention=multi_latent_attention,
                num_experts=(
                    transformer_config.num_moe_experts
                    if layer_uses_moe(transformer_config, layer_index)
                    else None
                ),
                moe_grouped_gemm=moe_grouped_gemm,
                is_mtp_layer=True,
                attention_spec=mtp_attention_spec,
            )
        )

    decoder = ModuleSpec(
        module=TransformerBlock,
        submodules=TransformerBlockSubmodules(
            layer_specs=layer_specs,
            layer_norm=_te_norm(role="final_rmsnorm"),
        ),
        metainfo={"backend": "transformer_engine"},
    )
    return ModuleSpec(
        module=GPTModel,
        submodules=GPTModelSubmodules(
            embedding=_spec(Embedding, role="input_embedding"),
            decoder=decoder,
            output_layer=_spec(TEColumnParallelLinear, role="lm_head"),
            vision_model=vision_model_spec,
            mtp=mtp_model_spec,
        ),
        metainfo={
            "backend": "transformer_engine",
            "model_type": model_config.model_type,
            "architectures": tuple(model_config.architectures),
        },
    )


def build_gpt_model_with_transformer_engine_spec(
    transformer_config,
    model_config,
    *,
    vision_config=None,
    moe_grouped_gemm=True,
):
    """Backward-compatible entry point routed through the model registry."""

    from types import SimpleNamespace

    from hcu_train_simulator.models.registry import resolve_model_adapter

    raw_identity = {
        "model_type": model_config.model_type,
        "architectures": list(model_config.architectures),
    }
    adapter = resolve_model_adapter(raw_identity)
    compatibility_config = SimpleNamespace(
        transformer=transformer_config,
        model=model_config,
        vision=vision_config or SimpleNamespace(enabled=False),
        parallel=SimpleNamespace(moe_grouped_gemm=moe_grouped_gemm),
    )
    return adapter.build_spec(compatibility_config)
