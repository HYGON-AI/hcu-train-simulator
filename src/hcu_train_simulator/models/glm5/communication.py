# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Register GLM-5 DSA compute as eligible communication-overlap work."""

from hcu_train_simulator.estimators.overlap_registry import register_overlap_parts


GLM5_DSA_ATTENTION_PARTS = {
    "dsa_index_q",
    "dsa_index_k",
    "dsa_index_weights",
    "dsa_index_k_norm",
    "dsa_index_rope",
    "dsa_index_score",
    "dsa_index_score_reduce",
    "dsa_topk",
    "dsa_sparse_attn",
    "dsa_attention_softmax",
}

register_overlap_parts(
    transformer=GLM5_DSA_ATTENTION_PARTS,
    attention=GLM5_DSA_ATTENTION_PARTS,
)
