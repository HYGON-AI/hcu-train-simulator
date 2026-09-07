# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Self-contained Megatron-Core style module specifications.

The simulator intentionally does not import Megatron-Core.  These classes mirror
the parts of its spec vocabulary that are needed to describe the training graph,
while the marker modules below remain lightweight semantic identifiers.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Union


# Semantic module markers.  They are never instantiated by the simulator; using
# types keeps ModuleSpec structurally compatible with Megatron-Core's API.
class GPTModel:
    pass


class TransformerBlock:
    pass


class TransformerLayer:
    pass


class SelfAttention:
    pass


class MLASelfAttention:
    pass


class DSASelfAttention:
    pass


class DSAIndexer:
    pass


class DSAIndexerScore:
    pass


class DSATopK:
    pass


class GatedSelfAttention:
    pass


class GatedDeltaNet:
    pass


class MultiTokenPredictor:
    pass


class MLP:
    pass


class MoELayer:
    pass


class Embedding:
    pass


class VisionModel:
    pass


class VisionTransformerBlock:
    pass


class VisionTransformerLayer:
    pass


class VisionSelfAttention:
    pass


class VisionMLP:
    pass


class VisionPatchEmbedding:
    pass


class VisionPositionEmbedding:
    pass


class VisionPatchMerger:
    pass


class ColumnParallelLinear:
    pass


class RowParallelLinear:
    pass


class TEDotProductAttention:
    pass


class TENorm:
    pass


class TEColumnParallelLinear:
    pass


class TELayerNormColumnParallelLinear:
    pass


class TERowParallelLinear:
    pass


class TEGroupedMLP:
    pass


class TESequentialMLP:
    pass


class TopKRouter:
    pass


class SharedExpertMLP:
    pass


class SwiGLU:
    pass


class GELU:
    pass


class SigmoidGate:
    pass


class CausalConv1d:
    pass


class GatedDeltaRule:
    pass


class GatedRMSNorm:
    pass


class BiasDropoutAdd:
    pass


class IdentityOp:
    pass


ModuleRef = Union[type, tuple[str, str]]


@dataclass(frozen=True)
class ModuleSpec:
    """Description of a module, its constructor params, and nested submodules."""

    module: ModuleRef
    params: dict[str, Any] = field(default_factory=dict)
    submodules: object = None
    metainfo: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SelfAttentionSubmodules:
    linear_qkv: ModuleSpec
    core_attention: ModuleSpec
    linear_proj: ModuleSpec
    q_layernorm: Optional[ModuleSpec] = None
    k_layernorm: Optional[ModuleSpec] = None


@dataclass(frozen=True)
class GatedSelfAttentionSubmodules:
    linear_q: ModuleSpec
    linear_k: ModuleSpec
    linear_v: ModuleSpec
    core_attention: ModuleSpec
    linear_proj: ModuleSpec
    q_layernorm: ModuleSpec
    k_layernorm: ModuleSpec
    output_gate: ModuleSpec


@dataclass(frozen=True)
class GatedDeltaNetSubmodules:
    in_proj_qkv: ModuleSpec
    in_proj_z: ModuleSpec
    in_proj_b: ModuleSpec
    in_proj_a: ModuleSpec
    conv1d: ModuleSpec
    core: ModuleSpec
    gated_norm: ModuleSpec
    out_proj: ModuleSpec


@dataclass(frozen=True)
class MLASelfAttentionSubmodules:
    linear_q_proj: Optional[ModuleSpec]
    linear_q_down_proj: Optional[ModuleSpec]
    linear_q_up_proj: ModuleSpec
    linear_kv_down_proj: ModuleSpec
    linear_kv_up_proj: ModuleSpec
    core_attention: ModuleSpec
    linear_proj: ModuleSpec
    q_layernorm: Optional[ModuleSpec] = None
    kv_layernorm: Optional[ModuleSpec] = None


@dataclass(frozen=True)
class DSAIndexerSubmodules:
    linear_q_proj: ModuleSpec
    linear_k_proj: ModuleSpec
    k_layernorm: ModuleSpec
    weights_proj: ModuleSpec
    score: ModuleSpec
    topk: ModuleSpec


@dataclass(frozen=True)
class DSASelfAttentionSubmodules:
    linear_q_down_proj: ModuleSpec
    linear_q_up_proj: ModuleSpec
    linear_kv_down_proj: ModuleSpec
    linear_kv_up_proj: ModuleSpec
    core_attention: ModuleSpec
    linear_proj: ModuleSpec
    q_layernorm: ModuleSpec
    kv_layernorm: ModuleSpec
    indexer: Optional[ModuleSpec]


@dataclass(frozen=True)
class MLPSubmodules:
    linear_fc1: ModuleSpec
    linear_fc2: ModuleSpec
    activation_func: Optional[ModuleSpec] = None


@dataclass(frozen=True)
class MoESubmodules:
    router: ModuleSpec
    experts: ModuleSpec
    shared_experts: Optional[ModuleSpec] = None


@dataclass(frozen=True)
class TransformerLayerSubmodules:
    input_layernorm: ModuleSpec
    self_attention: ModuleSpec
    self_attn_bda: ModuleSpec
    pre_mlp_layernorm: ModuleSpec
    mlp: ModuleSpec
    mlp_bda: ModuleSpec


@dataclass(frozen=True)
class TransformerBlockSubmodules:
    layer_specs: list[ModuleSpec]
    layer_norm: Optional[ModuleSpec] = None


@dataclass(frozen=True)
class MultiTokenPredictorSubmodules:
    embedding: ModuleSpec
    pre_fc_norm_embedding: ModuleSpec
    pre_fc_norm_hidden: ModuleSpec
    linear_fc: ModuleSpec
    norm: ModuleSpec


@dataclass(frozen=True)
class VisionAttentionSubmodules:
    linear_qkv: ModuleSpec
    core_attention: ModuleSpec
    linear_proj: ModuleSpec


@dataclass(frozen=True)
class VisionLayerSubmodules:
    input_layernorm: ModuleSpec
    self_attention: ModuleSpec
    self_attn_bda: ModuleSpec
    pre_mlp_layernorm: ModuleSpec
    mlp: ModuleSpec
    mlp_bda: ModuleSpec


@dataclass(frozen=True)
class VisionBlockSubmodules:
    layer_specs: list[ModuleSpec]


@dataclass(frozen=True)
class VisionModelSubmodules:
    patch_embedding: ModuleSpec
    position_embedding: ModuleSpec
    decoder: ModuleSpec
    merger: ModuleSpec
    deepstack_mergers: list[ModuleSpec] = field(default_factory=list)


@dataclass(frozen=True)
class GPTModelSubmodules:
    embedding: ModuleSpec
    decoder: ModuleSpec
    output_layer: ModuleSpec
    vision_model: Optional[ModuleSpec] = None
    mtp: Optional[ModuleSpec] = None


def module_is(spec: Optional[ModuleSpec], module: type) -> bool:
    return isinstance(spec, ModuleSpec) and spec.module is module


def walk_module_specs(spec: ModuleSpec):
    """Yield a spec tree without importing or instantiating its modules."""

    yield spec
    submodules = spec.submodules
    if submodules is None:
        return
    for value in vars(submodules).values():
        if isinstance(value, ModuleSpec):
            yield from walk_module_specs(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, ModuleSpec):
                    yield from walk_module_specs(item)


def iter_named_module_specs(spec: ModuleSpec, path: str = "model"):
    """Yield ``(path, spec)`` pairs for a ModuleSpec tree.

    Paths use submodule field names and list indexes, which makes estimator
    output traceable back to the exact ModuleSpec that produced it.
    """

    yield path, spec
    submodules = spec.submodules
    if submodules is None:
        return
    for name, value in vars(submodules).items():
        if isinstance(value, ModuleSpec):
            yield from iter_named_module_specs(value, f"{path}.{name}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, ModuleSpec):
                    yield from iter_named_module_specs(item, f"{path}.{name}[{index}]")


def module_type_name(spec: ModuleSpec) -> str:
    """Return a stable display name for local and external module refs."""

    if isinstance(spec.module, tuple):
        return ".".join(spec.module)
    return spec.module.__name__


def contains_module(spec: ModuleSpec, module: type) -> bool:
    return any(module_is(item, module) for item in walk_module_specs(spec))


def get_transformer_block_submodules(spec: ModuleSpec) -> TransformerBlockSubmodules:
    if module_is(spec, GPTModel):
        decoder = spec.submodules.decoder
        if not module_is(decoder, TransformerBlock):
            raise TypeError("GPTModel decoder must be a TransformerBlock ModuleSpec")
        return decoder.submodules
    if module_is(spec, TransformerBlock):
        return spec.submodules
    raise TypeError("expected a GPTModel or TransformerBlock ModuleSpec")


def get_layer_specs(spec: ModuleSpec, include_mtp: bool = True) -> list[ModuleSpec]:
    layers = get_transformer_block_submodules(spec).layer_specs
    if include_mtp:
        return list(layers)
    return [layer for layer in layers if not layer.metainfo.get("is_mtp_layer", False)]


def get_representative_layer_spec(spec: ModuleSpec) -> ModuleSpec:
    layers = get_layer_specs(spec, include_mtp=False)
    if not layers:
        raise ValueError("model spec has no transformer layers")
    return layers[0]


def get_attention_spec(layer_spec: ModuleSpec) -> ModuleSpec:
    if not module_is(layer_spec, TransformerLayer):
        raise TypeError("expected a TransformerLayer ModuleSpec")
    return layer_spec.submodules.self_attention


def get_mlp_spec(layer_spec: ModuleSpec) -> ModuleSpec:
    if not module_is(layer_spec, TransformerLayer):
        raise TypeError("expected a TransformerLayer ModuleSpec")
    return layer_spec.submodules.mlp


def spec_uses_mla(spec: ModuleSpec) -> bool:
    return contains_module(spec, MLASelfAttention) or contains_module(spec, DSASelfAttention)


def spec_uses_dsa(spec: ModuleSpec) -> bool:
    return contains_module(spec, DSASelfAttention)


def spec_uses_moe(spec: ModuleSpec) -> bool:
    return contains_module(spec, MoELayer)


def spec_has_qk_norm(spec: ModuleSpec) -> bool:
    for layer in get_layer_specs(spec, include_mtp=False):
        attention = get_attention_spec(layer)
        submodules = attention.submodules
        if module_is(attention, SelfAttention) or module_is(attention, GatedSelfAttention):
            if (
                submodules.q_layernorm is not None
                and not module_is(submodules.q_layernorm, IdentityOp)
            ) or (
                submodules.k_layernorm is not None
                and not module_is(submodules.k_layernorm, IdentityOp)
            ):
                return True
    return False


def spec_uses_gated_delta_net(spec: ModuleSpec) -> bool:
    return contains_module(spec, GatedDeltaNet)


def spec_uses_vision(spec: ModuleSpec) -> bool:
    return contains_module(spec, VisionModel)


def spec_uses_mtp(spec: ModuleSpec) -> bool:
    return contains_module(spec, MultiTokenPredictor)
