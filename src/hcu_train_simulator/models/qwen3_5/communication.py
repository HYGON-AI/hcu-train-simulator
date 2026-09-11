# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from hcu_train_simulator.estimators.overlap_registry import register_overlap_parts


QWEN35_ATTENTION_PARTS = {
    "qwen35_q_proj_gate", "qwen35_k_proj", "qwen35_v_proj", "attention_output_gate",
    "gdn_in_proj_qkv", "gdn_in_proj_z", "gdn_in_proj_a", "gdn_in_proj_b",
    "gdn_causal_conv1d", "gdn_gate_preprocess", "gdn_qk_l2norm",
    "gated_delta_rule", "gdn_gated_rmsnorm", "gdn_out_proj",
}
QWEN35_MTP_PARTS = {
    "mtp_pre_fc_norm_embedding", "mtp_pre_fc_norm_hidden", "mtp_fc", "mtp_final_norm",
}

register_overlap_parts(
    transformer=QWEN35_ATTENTION_PARTS | QWEN35_MTP_PARTS,
    attention=QWEN35_ATTENTION_PARTS,
)
