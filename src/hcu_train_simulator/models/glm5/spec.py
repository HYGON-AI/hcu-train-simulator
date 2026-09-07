# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""GLM-5 MLA + DeepSeek Sparse Attention module specifications."""

from hcu_train_simulator.modeling.spec import (
    DSAIndexer,
    DSAIndexerScore,
    DSAIndexerSubmodules,
    DSASelfAttention,
    DSASelfAttentionSubmodules,
    DSATopK,
    ModuleSpec,
    TEColumnParallelLinear,
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)


def _spec(module, *, role=None, **params):
    metainfo = {"role": role} if role else {}
    return ModuleSpec(module=module, params=params, metainfo=metainfo)


def build_dsa_attention_spec(*, include_indexer=True):
    """Describe GLM-5's MLA path and its replicated lightning indexer."""

    indexer = None
    if include_indexer:
        indexer = ModuleSpec(
            module=DSAIndexer,
            submodules=DSAIndexerSubmodules(
                linear_q_proj=_spec(TEColumnParallelLinear, role="dsa_index_q"),
                linear_k_proj=_spec(TEColumnParallelLinear, role="dsa_index_k"),
                k_layernorm=_spec(
                    TENorm,
                    role="dsa_index_k_norm",
                    normalization="LayerNorm",
                    scope="index_head_dim",
                ),
                weights_proj=_spec(TEColumnParallelLinear, role="dsa_index_weights"),
                score=_spec(DSAIndexerScore, role="dsa_index_score"),
                topk=_spec(DSATopK, role="dsa_topk"),
            ),
            metainfo={"backend": "dsa", "replicated_over_tp": True},
        )

    return ModuleSpec(
        module=DSASelfAttention,
        params={"attn_mask_type": "causal"},
        submodules=DSASelfAttentionSubmodules(
            linear_q_down_proj=_spec(
                TELayerNormColumnParallelLinear,
                role="mla_q_down",
                fused_norm_scope="hidden_size",
            ),
            linear_q_up_proj=_spec(TEColumnParallelLinear, role="mla_q_up"),
            linear_kv_down_proj=_spec(
                TELayerNormColumnParallelLinear,
                role="mla_kv_down",
            ),
            linear_kv_up_proj=_spec(TEColumnParallelLinear, role="mla_kv_up"),
            core_attention=_spec(TEDotProductAttention, role="dsa_sparse_attn"),
            linear_proj=_spec(TERowParallelLinear, role="attn_proj"),
            q_layernorm=_spec(
                TENorm,
                role="mla_q_layernorm",
                normalization="RMSNorm",
                scope="q_lora_rank",
            ),
            kv_layernorm=_spec(
                TENorm,
                role="mla_kv_layernorm",
                normalization="RMSNorm",
                scope="kv_lora_rank",
            ),
            indexer=indexer,
        ),
        metainfo={
            "backend": "transformer_engine",
            "attention_variant": "dsa",
            "indexer_mode": "full" if include_indexer else "shared",
        },
    )
