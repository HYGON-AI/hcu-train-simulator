# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""GLM-5 DSA analytical FLOP registration."""

from hcu_train_simulator.modeling.architecture import (
    attention_output_dim,
    attention_score_dim,
    kv_projection_dim,
    query_projection_dim,
    value_head_dim,
)
from hcu_train_simulator.modeling.spec import DSASelfAttention
from hcu_train_simulator.modeling.statistics_registry import register_attention_flops


@register_attention_flops(DSASelfAttention)
def dsa_attention_flops(model, attention, seq, train_matmul):
    hidden = model.hidden_size
    q_rank = model.q_lora_rank
    mla_weights = (
        hidden * q_rank
        + q_rank * query_projection_dim(model)
        + hidden * (model.kv_lora_rank + model.qk_rope_head_dim)
        + model.kv_lora_rank * kv_projection_dim(model)
        + attention_output_dim(model) * hidden
    )
    indexer_weights = 0
    indexer_score = 0
    if attention.submodules.indexer is not None:
        indexer_weights = (
            q_rank * model.index_n_heads * model.index_head_dim
            + hidden * model.index_head_dim
            + hidden * model.index_n_heads
        )
        indexer_score = (
            train_matmul
            * model.index_n_heads
            * model.index_head_dim
            * seq
        )
    sparse_k = min(model.index_topk, seq)
    sparse_attention = (
        train_matmul
        * model.num_attention_heads
        * (attention_score_dim(model) + value_head_dim(model))
        * sparse_k
    )
    return (
        train_matmul * (mla_weights + indexer_weights)
        + indexer_score
        + sparse_attention
    )
