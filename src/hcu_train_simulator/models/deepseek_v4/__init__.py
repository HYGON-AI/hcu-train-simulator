# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek V4 and V4.1 training architecture adapters."""

from hcu_train_simulator.models.base import (
    ModelAdapter,
    NormalizedModelInput,
    merge_language_metadata,
    with_rmsnorm_defaults,
)
from hcu_train_simulator.models.registry import register_model_adapter

from . import costs  # noqa: F401
from .communication import context_parallel_time, engram_parallel_time, pipeline_activation_width
from .spec import build_model_spec


class DeepSeekV4Adapter(ModelAdapter):
    name = "deepseek_v4"
    model_types = ("deepseek_v4",)
    model_type_prefixes = ("deepseek_v4_",)
    architecture_prefixes = ("DeepseekV4", "DeepSeekV4")
    csa2 = False

    def normalize_config(self, raw_config):
        text = merge_language_metadata(raw_config.get("text_config", raw_config), raw_config)
        text = with_rmsnorm_defaults(text)
        text.setdefault("num_key_value_heads", 1)
        text["moe_shared_expert_gate"] = False
        layers = text["num_hidden_layers"]
        if layers <= 0:
            raise ValueError("DeepSeek V4 requires at least one backbone layer")
        mtp_layers = next(
            (text[key] for key in (
                "mtp_num_layers", "num_mtp_layers", "num_nextn_predict_layers",
                "num_nextn_predict_layer", "mtp_num_hidden_layers",
            ) if text.get(key) is not None),
            0,
        )
        ratios = text.get("compress_ratios")
        if ratios is None and not self.csa2:
            rates = {
                "sliding_attention": 0,
                "compressed_sparse_attention": text.get("compress_rate_csa", 4),
                "heavily_compressed_attention": text.get("compress_rate_hca", 128),
                **(text.get("compress_rates") or {}),
            }
            try:
                ratios = [rates[kind] for kind in text["layer_types"]]
            except KeyError as exc:
                raise ValueError("DeepSeek V4 requires compress_ratios or valid layer_types") from exc
        if ratios is None or len(ratios) not in {layers, layers + mtp_layers}:
            raise ValueError("compress_ratios must cover the backbone, optionally followed by MTP layers")
        rates = text.get("compress_rates") or {}
        csa_ratio = rates.get("compressed_sparse_attention", text.get("compress_rate_csa", 4))
        hca_ratio = rates.get("heavily_compressed_attention", text.get("compress_rate_hca", 128))
        if not self.csa2 and (min(csa_ratio, hca_ratio) < 2 or csa_ratio == hca_ratio):
            raise ValueError("V4 CSA and HCA require distinct compression ratios greater than one")
        supported_ratios = {0, 1, 2} if self.csa2 else {0, csa_ratio, hca_ratio}
        if any(not isinstance(ratio, int) or ratio not in supported_ratios for ratio in ratios):
            raise ValueError(f"invalid DeepSeek compression ratios (supported: {sorted(supported_ratios)})")
        if any(ratios[layers:]):
            raise ValueError("DeepSeek V4 auxiliary layers require sliding-window attention")

        options = {
            "csa2": self.csa2,
            "ratios": tuple(ratios[:layers]),
            "csa_ratio": csa_ratio,
            "window": text.get("sliding_window", 128),
            "o_groups": text["o_groups"],
            "o_rank": text["o_lora_rank"],
            "hc_mult": text.get("hc_mult", 4),
            "hc_iters": text.get("hc_sinkhorn_iters", 20),
            "hash_layers": self.hash_layers(text, layers),
        }
        if min(options[key] for key in ("window", "o_groups", "o_rank", "hc_mult", "hc_iters")) <= 0:
            raise ValueError("DeepSeek window, output projection and mHC dimensions must be positive")
        if not any(text.get(key) for key in ("n_routed_experts", "num_experts", "num_local_experts")):
            raise ValueError("DeepSeek V4 requires routed MoE experts")
        if (
            text.get("first_k_dense_replace") or text.get("moe_dense_layer_indices")
            or text.get("moe_layer_offset") or text.get("moe_layer_freq", 1) != 1
        ):
            raise ValueError("DeepSeek V4 uses a MoE FFN on every layer")
        vision = raw_config.get("vision_config")
        if self.csa2:
            # The released checkpoint includes DSpark under mtp.*, but the
            # backbone trains without the draft objective.
            mtp_layers = 0
            options.update(self.csa2_options(text, ratios[:layers]))
            if vision is not None:
                vision = dict(vision)
                vision["spatial_merge_size"] = vision["downsample_ratio"]
                vision["num_position_embeddings"] = (
                    vision["max_image_tokens"] * vision["downsample_ratio"] ** 2
                )
        text["mtp_num_layers"] = mtp_layers
        text["architecture_options"] = options
        return NormalizedModelInput(language=text, vision=vision)

    def hash_layers(self, text, layers):
        mlp_types = text.get("mlp_layer_types")
        if mlp_types is not None:
            if len(mlp_types) != layers or set(mlp_types) - {"moe", "hash_moe"}:
                raise ValueError("mlp_layer_types must have one moe/hash_moe entry per backbone layer")
            return frozenset(i for i, kind in enumerate(mlp_types) if kind == "hash_moe")
        count = text.get("num_hash_layers", 0 if self.csa2 else min(3, layers))
        if not 0 <= count <= layers:
            raise ValueError("num_hash_layers must be between zero and num_hidden_layers")
        return frozenset(range(count))

    def build_spec(self, config):
        return build_model_spec(config)

    def validate(self, config):
        t = config.transformer
        p = config.parallel
        options = t.architecture_options
        if not t.num_moe_experts or not t.moe_router_topk or not t.moe_ffn_hidden_size:
            raise ValueError("DeepSeek V4 requires routed MoE dimensions")
        if t.num_query_groups != 1:
            raise ValueError("DeepSeek V4 uses one shared KV latent")
        if t.num_attention_heads % options["o_groups"] or options["o_groups"] % p.tp_size:
            raise ValueError("o_groups must divide num_attention_heads and be divisible by tp_size")
        if p.seq_length <= 0 or p.seq_length % p.cp_size:
            raise ValueError("seq_length must be positive and divisible by cp_size")
        if (t.q_lora_rank or 0) <= 0 or not 0 <= (t.qk_rope_head_dim or 0) <= t.kv_channels:
            raise ValueError("DeepSeek V4 requires a positive q_lora_rank and valid RoPE dimensions")
        if any(options["ratios"]) and (self.csa2 or options["csa_ratio"] in options["ratios"]):
            if min(t.index_topk, t.index_n_heads, t.index_head_dim) <= 0:
                raise ValueError("compressed sparse attention requires positive indexer dimensions")

    def pipeline_activation_width(self, config):
        return pipeline_activation_width(config)

    def context_parallel_time(self, simulator, config, num_microbatches):
        return context_parallel_time(simulator, config, num_microbatches)

    def extra_expert_parallel_time(self, simulator, config, num_microbatches):
        return engram_parallel_time(simulator, config, num_microbatches)


class DeepSeekV41Adapter(DeepSeekV4Adapter):
    name = "deepseek_v41"
    model_types = ("deepseek_v41", "deepseek_v41_text", "deepseek_v4_1", "deepseek_v4_1_text")
    model_type_prefixes = ("deepseek_v41_", "deepseek_v4_1_")
    architecture_prefixes = ("DeepseekV41", "DeepSeekV41", "DeepseekV4_1", "DeepSeekV4_1")
    csa2 = True

    def csa2_options(self, text, ratios):
        kv_sources = frozenset(text.get("kv_source_layer_ids", ()))
        index_sources = frozenset(text.get("index_source_layer_ids", ()))
        active = {i for i, ratio in enumerate(ratios) if ratio}
        if not kv_sources <= index_sources <= active:
            raise ValueError("CSA2 requires KV source layers to be index sources with nonzero ratios")
        previous_ratio = None
        for i, ratio in enumerate(ratios):
            if ratio and i not in kv_sources and ratio != previous_ratio:
                raise ValueError(f"CSA2 layer {i} has no preceding KV source of ratio {ratio}")
            if i in kv_sources:
                previous_ratio = ratio

        engram_layers = text.get("engram_layer_ids", ())
        rows = text.get("engram_num_embeddings", ())
        if (
            len(engram_layers) != len(rows)
            or len(set(engram_layers)) != len(rows)
            or any(not 0 <= layer < len(ratios) for layer in engram_layers)
            or any(size <= 0 for size in rows)
        ):
            raise ValueError("Engram requires one positive table size per distinct backbone layer")
        lookup_heads = (text.get("engram_max_ngram_size", 0) - 1) * text.get("engram_n_heads", 0)
        head_dim = text.get("engram_head_dim", 0)
        if rows and min(lookup_heads, head_dim) <= 0:
            raise ValueError("Engram requires positive hash and embedding dimensions")

        # Candidate restriction was introduced in post-training.
        candidate_source = text.get("candidate_source_layer_id", -1)
        candidate_blocks = text.get("candidate_topk_blocks", 0)
        block_size = text.get("candidate_block_size", 0)
        use_candidates = text.get("use_hierarchical_indexer", False)
        if use_candidates and (candidate_source not in kv_sources or min(candidate_blocks, block_size) <= 0):
            raise ValueError("hierarchical indexing requires a candidate KV source and positive block dimensions")
        return {
            "kv_sources": kv_sources,
            "index_sources": index_sources,
            "candidate_source": candidate_source if use_candidates else -1,
            "candidate_blocks": candidate_blocks,
            "candidate_block_size": block_size,
            "engram_rows": dict(zip(engram_layers, rows)),
            "engram_lookup_heads": lookup_heads,
            "engram_head_dim": head_dim,
        }


register_model_adapter(DeepSeekV4Adapter())
register_model_adapter(DeepSeekV41Adapter())
