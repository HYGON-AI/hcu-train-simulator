# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from hcu_train_simulator.modeling.architecture import round_up
from hcu_train_simulator.modeling.spec import ModuleSpec, spec_uses_vision
from hcu_train_simulator.models.registry import get_model_adapter, resolve_model_adapter


def _first_present(config, *keys, default=None):
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return default


def _text_config(config):
    """Return the language sub-config while preserving top-level HF metadata."""

    nested = config.get("text_config")
    if not isinstance(nested, dict):
        return config
    merged = dict(nested)
    for key in ("architectures", "model_type", "tie_word_embeddings"):
        if key in config:
            merged[key] = config[key]
    return merged


@dataclass(frozen=True)
class ModelConfig:
    """Language-model fields that live outside Megatron TransformerConfig."""

    original_vocab_size: int
    vocab_size: int
    vocab_padding_multiple: Optional[int]
    tie_word_embeddings: bool
    architectures: list
    model_type: Optional[str]

    @classmethod
    def from_dict(cls, config):
        config = _text_config(config)
        original_vocab_size = config['vocab_size']
        vocab_padding_multiple = _first_present(
            config,
            "make_vocab_size_divisible_by",
            "vocab_size_multiple",
            "padded_vocab_multiple",
        )
        padded_vocab_size = _first_present(
            config,
            "padded_vocab_size",
            "padded_vocab_size_with_padding",
        )
        if padded_vocab_size is None:
            padded_vocab_size = round_up(original_vocab_size, vocab_padding_multiple)

        return cls(
            original_vocab_size=original_vocab_size,
            vocab_size=padded_vocab_size,
            vocab_padding_multiple=vocab_padding_multiple,
            tie_word_embeddings=config.get('tie_word_embeddings', False),
            architectures=config.get('architectures', []),
            model_type=config.get("model_type"),
        )


@dataclass(frozen=True)
class VisionConfig:
    """Normalized visual-encoder fields consumed by multimodal adapters."""

    enabled: bool = False
    num_layers: int = 0
    hidden_size: int = 0
    num_attention_heads: int = 0
    ffn_hidden_size: int = 0
    in_channels: int = 3
    patch_size: int = 0
    temporal_patch_size: int = 0
    spatial_merge_size: int = 1
    output_hidden_size: int = 0
    num_position_embeddings: int = 0
    deepstack_visual_indexes: tuple[int, ...] = ()

    @classmethod
    def from_dict(cls, config):
        vision = config.get("vision_config")
        if not isinstance(vision, dict):
            return cls()
        patch_size = vision.get("patch_size", 0)
        temporal_patch_size = vision.get("temporal_patch_size", 0)
        if isinstance(patch_size, (list, tuple)):
            patch_size = patch_size[0]
        if isinstance(temporal_patch_size, (list, tuple)):
            temporal_patch_size = temporal_patch_size[0]
        return cls(
            enabled=True,
            num_layers=vision.get("depth", vision.get("num_hidden_layers", 0)),
            hidden_size=vision["hidden_size"],
            num_attention_heads=vision.get("num_heads", vision.get("num_attention_heads", 0)),
            ffn_hidden_size=vision["intermediate_size"],
            in_channels=vision.get("in_channels", 3),
            patch_size=int(patch_size),
            temporal_patch_size=int(temporal_patch_size),
            spatial_merge_size=vision.get("spatial_merge_size", 1),
            output_hidden_size=vision.get("out_hidden_size", _text_config(config).get("hidden_size", 0)),
            num_position_embeddings=vision.get("num_position_embeddings", 0),
            deepstack_visual_indexes=tuple(vision.get("deepstack_visual_indexes") or ()),
        )


@dataclass(frozen=True)
class TransformerConfig:
    """Megatron-Core aligned transformer architecture configuration.

    Simulator-only MLA, shared-expert, and MTP extensions remain explicit while
    the common field names follow Megatron-Core rather than HuggingFace.
    """

    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_query_groups: int
    kv_channels: int
    ffn_hidden_size: int
    num_moe_experts: Optional[int]
    moe_router_topk: Optional[int]
    moe_ffn_hidden_size: Optional[int]
    num_shared_experts: Optional[int]
    shared_expert_ffn_hidden_size: Optional[int]
    first_k_dense_replace: int
    moe_layer_freq: int
    moe_layer_offset: int
    moe_dense_layer_indices: tuple[int, ...]
    q_lora_rank: Optional[int]
    kv_lora_rank: Optional[int]
    qk_nope_head_dim: Optional[int]
    qk_rope_head_dim: Optional[int]
    v_head_dim: Optional[int]
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    indexer_types: tuple[str, ...]
    mtp_num_layers: int
    normalization: str
    layernorm_epsilon: float
    gated_linear_unit: bool
    add_qkv_bias: bool
    layer_types: tuple[str, ...]
    full_attention_interval: int
    attention_output_gate: bool
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    moe_shared_expert_gate: bool
    partial_rotary_factor: float
    mtp_use_dedicated_embeddings: bool

    @classmethod
    def from_dict(cls, config):
        config = _text_config(config)
        moe_ffn_hidden_size = _first_present(config, "moe_intermediate_size", "moe_ffn_hidden_size")
        rmsnorm_family = (
            "rms_norm_eps" in config
            or str(config.get("normalization", "")).lower() == "rmsnorm"
        )
        return cls(
            num_layers=config["num_hidden_layers"],
            hidden_size=config["hidden_size"],
            num_attention_heads=config["num_attention_heads"],
            num_query_groups=_first_present(
                config,
                "num_key_value_heads",
                "num_kv_heads",
                default=config["num_attention_heads"],
            ),
            kv_channels=config.get("head_dim", config["hidden_size"] // config["num_attention_heads"]),
            ffn_hidden_size=_first_present(
                config,
                "intermediate_size",
                "ffn_hidden_size",
                default=moe_ffn_hidden_size,
            ),
            num_moe_experts=_first_present(config, "num_experts", "n_routed_experts"),
            moe_router_topk=_first_present(
                config,
                "num_experts_per_tok",
                "num_experts_per_token",
                "topk",
            ),
            moe_ffn_hidden_size=moe_ffn_hidden_size,
            num_shared_experts=_first_present(
                config,
                "num_shared_experts",
                "n_shared_experts",
                default=(1 if config.get("shared_expert_intermediate_size") else None),
            ),
            shared_expert_ffn_hidden_size=_first_present(
                config,
                "shared_expert_intermediate_size",
                "shared_expert_ffn_hidden_size",
                "moe_shared_expert_intermediate_size",
                default=moe_ffn_hidden_size,
            ),
            first_k_dense_replace=int(config.get("first_k_dense_replace", 0) or 0),
            moe_layer_freq=int(config.get("moe_layer_freq", 1)),
            moe_layer_offset=int(config.get("moe_layer_offset", 0) or 0),
            moe_dense_layer_indices=tuple(config.get("moe_dense_layer_indices") or ()),
            q_lora_rank=_first_present(config, "q_lora_rank"),
            kv_lora_rank=_first_present(config, "kv_lora_rank"),
            qk_nope_head_dim=_first_present(config, "qk_nope_head_dim"),
            qk_rope_head_dim=_first_present(config, "qk_rope_head_dim"),
            v_head_dim=_first_present(config, "v_head_dim"),
            index_n_heads=int(config.get("index_n_heads", 0) or 0),
            index_head_dim=int(config.get("index_head_dim", 0) or 0),
            index_topk=int(config.get("index_topk", 0) or 0),
            indexer_types=tuple(config.get("indexer_types") or ()),
            mtp_num_layers=_first_present(
                config,
                "mtp_num_layers",
                "num_mtp_layers",
                "num_nextn_predict_layers",
                "num_nextn_predict_layer",
                "mtp_num_hidden_layers",
                default=0,
            ),
            normalization=(
                "RMSNorm"
                if "rms_norm_eps" in config or rmsnorm_family
                else config.get("normalization", "LayerNorm")
            ),
            layernorm_epsilon=_first_present(
                config,
                "rms_norm_eps",
                "layer_norm_eps",
                default=1e-6 if rmsnorm_family else 1e-5,
            ),
            gated_linear_unit=config.get("hidden_act", "silu").lower() in {"silu", "swiglu"},
            add_qkv_bias=config.get("attention_bias", config.get("add_qkv_bias", False)),
            layer_types=tuple(config.get("layer_types") or ()),
            full_attention_interval=int(config.get("full_attention_interval", 4)),
            attention_output_gate=bool(config.get("attn_output_gate", config.get("attention_output_gate", False))),
            linear_conv_kernel_dim=int(config.get("linear_conv_kernel_dim", 0) or 0),
            linear_key_head_dim=int(config.get("linear_key_head_dim", 0) or 0),
            linear_value_head_dim=int(config.get("linear_value_head_dim", 0) or 0),
            linear_num_key_heads=int(config.get("linear_num_key_heads", 0) or 0),
            linear_num_value_heads=int(config.get("linear_num_value_heads", 0) or 0),
            moe_shared_expert_gate=bool(config.get("moe_shared_expert_gate", bool(config.get("shared_expert_intermediate_size")))),
            partial_rotary_factor=float(
                (config.get("rope_parameters") or {}).get(
                    "partial_rotary_factor",
                    config.get("partial_rotary_factor", 1.0),
                )
            ),
            mtp_use_dedicated_embeddings=bool(
                config.get("mtp_use_dedicated_embeddings", False)
            ),
        )


@dataclass
class ParallelConfig:
    # parallel config
    tp_size: int
    cp_size: int
    pp_size: int
    ep_size: int
    etp_size: Optional[int]
    num_layers_per_vp_stage: Optional[int]
    use_distributed_optimizer: bool
    full_recompute: bool
    decoder_first_pipeline_num_layers: Optional[int]
    decoder_last_pipeline_num_layers: Optional[int]

    # training config
    micro_batch_size: int
    global_batch_size: int
    seq_length: int
    num_gpus: int

    # overlap ratio
    tp_overlap_ratio: Optional[float]
    dp_overlap_ratio: Optional[float]
    ep_overlap_ratio: Optional[float]
    pp_overlap_ratio: Optional[float]
    cp_overlap_ratio: Optional[float] = 0.0
    etp_overlap_ratio: Optional[float] = 0.0
    edp_overlap_ratio: Optional[float] = 0.0
    overlap_mode: str = "auto"
    # Expert-parallel dispatch/combine overlap depends on framework support and
    # must not be assumed merely because an expert compute window exists.
    ep_overlap_enabled: bool = False
    # Megatron P2P overlap is only available with an interleaved VPP schedule.
    overlap_p2p_comm: bool = False
    ep_communication_backend: str = "alltoall"

    # fp8 training
    # switch only the explicitly modeled fp8 training compute/communication paths.
    # memory and benchmark estimators remain unchanged.
    use_fp8_training: bool = False

    # derived field
    dp_size: int = field(init=False)
    # vp_size: int = field(init=False)
    vp_size: int = -1
    pp_schedule: str = "1f1b"
    sequence_parallel: bool = True

    # operator benchmark config
    moe_grouped_gemm: bool = True
    benchmark_dtype: str = "bf16"
    benchmark_device: str = "cuda"
    benchmark_allow_cpu: bool = False
    benchmark_warmup: int = 5
    benchmark_iters: int = 20
    benchmark_ops: list[str] = field(default_factory=lambda: ["all"])
    benchmark_dropout_p: float = 0.0
    benchmark_causal: bool = True
    benchmark_attention_kv_context: str = "local"
    benchmark_optimizer_numel: int = 16_777_216
    benchmark_max_local_experts: Optional[int] = None
    benchmark_qk_norm_scale: float = 0.4
    vision_seq_length: Optional[int] = None
    vision_num_images: int = 1

    @classmethod
    def from_dict(cls, config):
        return cls(**config)


@dataclass
class HardwareConfig:
    fp16_tflops: float
    fp8_tflops: float
    gpus_per_node: int
    hbm_gib: float
    intra_bw_gbps: float
    inter_bw_gbps: float
    gemm_efficiency: float
    p2p_intra_efficiency: float
    collective_intra_efficiency: float
    collective_inter_efficiency: float
    non_gemm_efficiency: float = 0.08
    optimizer_efficiency: float = 0.04
    use_bandwidth_table: bool = True
    # Resolved communication fields. New YAML should use the explicit
    # mode-specific names handled by from_dict; these fields keep downstream
    # code and legacy YAML compatible.
    use_supernode: bool = False
    nodes_per_supernode: int = 1
    scale_up_bw_gbps: Optional[float] = None
    collective_scale_up_efficiency: float = 1.0
    intra_node_num_gpus: Optional[int] = None
    intra_node_efficiency: Optional[float] = None
    scale_up_1_num_gpus: Optional[int] = None
    scale_up_1_bw_gbps: Optional[float] = None
    scale_up_1_efficiency: Optional[float] = None
    scale_up_2_num_gpus: Optional[int] = None
    scale_up_2_bw_gbps: Optional[float] = None
    scale_up_2_efficiency: Optional[float] = None
    scale_out_bw_gbps: Optional[float] = None
    scale_out_efficiency: Optional[float] = None

    @classmethod
    def from_dict(cls, config):
        use_supernode = bool(config.get("use_supernode", False))
        standard_keys = {
            "intra_node_num_gpus",
            "intra_node_bw_gbps",
            "intra_node_efficiency",
            "gpus_per_node",
            "intra_bw_gbps",
            "collective_intra_efficiency",
        }
        supernode_keys = {
            "scale_up_1_num_gpus",
            "scale_up_1_bw_gbps",
            "scale_up_1_efficiency",
            "scale_up_2_num_gpus",
            "scale_up_2_bw_gbps",
            "scale_up_2_efficiency",
            "nodes_per_supernode",
            "scale_up_bw_gbps",
            "collective_scale_up_efficiency",
        }
        inactive_keys = standard_keys if use_supernode else supernode_keys
        configured_inactive = sorted(key for key in inactive_keys if key in config)
        if configured_inactive:
            active_mode = "supernode" if use_supernode else "standard-node"
            raise ValueError(
                f"hardware_config selects {active_mode} mode but also configures "
                f"inactive fields: {', '.join(configured_inactive)}"
            )
        scale_out_bw = config.get("scale_out_bw_gbps", config.get("inter_bw_gbps"))
        scale_out_efficiency = config.get(
            "scale_out_efficiency",
            config.get("collective_inter_efficiency"),
        )
        p2p_intra_efficiency = config.get("p2p_intra_efficiency")

        common = dict(
            fp16_tflops=config["fp16_tflops"],
            fp8_tflops=config["fp8_tflops"],
            hbm_gib=config["hbm_gib"],
            gemm_efficiency=config["gemm_efficiency"],
            non_gemm_efficiency=config.get("non_gemm_efficiency", 0.08),
            optimizer_efficiency=config.get("optimizer_efficiency", 0.04),
            use_bandwidth_table=config.get("use_bandwidth_table", True),
            use_supernode=use_supernode,
            inter_bw_gbps=scale_out_bw or 0,
            collective_inter_efficiency=scale_out_efficiency or 0,
            p2p_intra_efficiency=p2p_intra_efficiency or 0,
            scale_out_bw_gbps=scale_out_bw,
            scale_out_efficiency=scale_out_efficiency,
        )

        if use_supernode:
            level1_size = config.get("scale_up_1_num_gpus")
            level2_size = config.get("scale_up_2_num_gpus")
            level1_bw = config.get("scale_up_1_bw_gbps")
            level2_bw = config.get("scale_up_2_bw_gbps")
            level1_efficiency = config.get("scale_up_1_efficiency")
            level2_efficiency = config.get("scale_up_2_efficiency")
            nodes_per_supernode = (
                level2_size // level1_size
                if level1_size and level2_size and level2_size % level1_size == 0
                else 0
            )
            return cls(
                **common,
                gpus_per_node=level1_size or 0,
                intra_bw_gbps=level1_bw or 0,
                collective_intra_efficiency=level1_efficiency or 0,
                nodes_per_supernode=nodes_per_supernode,
                scale_up_bw_gbps=level2_bw,
                collective_scale_up_efficiency=level2_efficiency or 0,
                scale_up_1_num_gpus=level1_size,
                scale_up_1_bw_gbps=level1_bw,
                scale_up_1_efficiency=level1_efficiency,
                scale_up_2_num_gpus=level2_size,
                scale_up_2_bw_gbps=level2_bw,
                scale_up_2_efficiency=level2_efficiency,
            )

        intra_size = config.get("intra_node_num_gpus", config.get("gpus_per_node"))
        intra_bw = config.get("intra_node_bw_gbps", config.get("intra_bw_gbps"))
        intra_efficiency = config.get(
            "intra_node_efficiency",
            config.get("collective_intra_efficiency"),
        )
        return cls(
            **common,
            gpus_per_node=intra_size or 0,
            intra_bw_gbps=intra_bw or 0,
            collective_intra_efficiency=intra_efficiency or 0,
            intra_node_num_gpus=intra_size,
            intra_node_efficiency=intra_efficiency,
        )


@dataclass
class SearchConfig:
    enabled: bool = False
    top_k: int = 10
    memory_margin_gib: float = 0
    candidates: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config):
        config = config or {}
        return cls(
            enabled=config.get("enabled", False),
            top_k=config.get("top_k", 10),
            memory_margin_gib=config.get("memory_margin_gib", 0),
            candidates=config.get("candidates") or {},
        )


@dataclass
class ProfileConfig:
    enabled: bool = True
    mode: str = "auto"
    user_dir: str = "profiles/user"
    legacy_path: Optional[str] = None
    builtin_accelerator: Optional[str] = None
    builtin_torch: Optional[str] = None
    builtin_dtk: Optional[str] = None
    update_on_benchmark: bool = True
    require_idle: bool = True
    ignore_gemm_layout: bool = True
    match_permuted_gemm: bool = True

    @classmethod
    def from_dict(cls, config):
        config = config or {}
        mode = str(config.get("mode", "auto")).lower()
        selector = config.get("builtin_selector") or {}
        config_dir = config.get("_config_dir")

        def resolve_path(value):
            if not value:
                return None
            path = Path(value)
            if not path.is_absolute() and config_dir:
                path = Path(config_dir) / path
            return str(path.resolve())

        return cls(
            enabled=config.get("enabled", True),
            mode=mode,
            user_dir=resolve_path(config.get("user_dir") or "profiles/user"),
            legacy_path=resolve_path(config.get("path")),
            builtin_accelerator=selector.get("accelerator"),
            builtin_torch=(str(selector["torch"]) if selector.get("torch") is not None else None),
            builtin_dtk=(str(selector["dtk"]) if selector.get("dtk") is not None else None),
            update_on_benchmark=config.get("update_on_benchmark", True),
            require_idle=config.get("require_idle", True),
            ignore_gemm_layout=config.get("ignore_gemm_layout", True),
            match_permuted_gemm=config.get("match_permuted_gemm", True),
        )


@dataclass
class SimulationConfig:
    model: ModelConfig
    transformer: TransformerConfig
    vision: VisionConfig = field(metadata={"flatten": False})
    parallel: ParallelConfig
    hardware: HardwareConfig
    search: SearchConfig = field(default_factory=SearchConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    model_adapter: str = field(default="generic_gpt", metadata={"flatten": False})
    model_spec: ModuleSpec = field(init=False, metadata={"flatten": False})

    @classmethod
    def from_dict(cls, model_config, estimator_config):
        adapter = resolve_model_adapter(model_config)
        normalized = adapter.normalize_config(model_config)
        normalized_model = dict(normalized.language)
        if normalized.vision is not None:
            normalized_model["vision_config"] = normalized.vision
        return cls(
            model=ModelConfig.from_dict(normalized_model),
            transformer=TransformerConfig.from_dict(normalized_model),
            vision=VisionConfig.from_dict(normalized_model),
            parallel=ParallelConfig.from_dict(estimator_config["parallel_config"]),
            hardware=HardwareConfig.from_dict(estimator_config["hardware_config"]),
            search=SearchConfig.from_dict(estimator_config.get("search")),
            profile=ProfileConfig.from_dict(
                {
                    **(estimator_config.get("profile_config") or {}),
                    "_config_dir": estimator_config.get("_config_dir"),
                }
            ),
            model_adapter=adapter.name,
        )

    def __post_init__(self):
        self.parallel.dp_size = self.parallel.num_gpus // (self.parallel.tp_size * self.parallel.cp_size * self.parallel.pp_size)
        if self.parallel.num_layers_per_vp_stage:
            self.parallel.vp_size = self.transformer.num_layers // self.parallel.pp_size // self.parallel.num_layers_per_vp_stage

        if not self.parallel.etp_size:
            if self.transformer.num_moe_experts:
                # moe模型设置为tp size, 与megatron保持一致
                self.parallel.etp_size = self.parallel.tp_size
            else:
                # dense模型设置为1避免通信模拟报错
                self.parallel.etp_size = 1

        self.model_spec = get_model_adapter(self.model_adapter).build_spec(self)
        if self.vision.enabled and not spec_uses_vision(self.model_spec):
            raise ValueError(
                f"model adapter {self.model_adapter!r} accepted a vision_config "
                "but built a text-only ModuleSpec; refusing to drop the vision model"
            )
        if not self.vision.enabled and spec_uses_vision(self.model_spec):
            raise ValueError(
                f"model adapter {self.model_adapter!r} built a vision ModuleSpec "
                "without a vision_config"
            )
