# Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""ModuleSpec-driven parameter and activation estimators.

The public estimators aggregate a ModuleSpec tree, while every non-zero cost is
owned by a concrete node and can be inspected through ``memory_rows``.  This
keeps architecture selection in the spec builder instead of duplicating model
family conditionals in the memory estimator.
"""

from dataclasses import dataclass

from hcu_train_simulator.config.utils import flatten_dataclass
from hcu_train_simulator.modeling.architecture import (
    attention_output_dim,
    kv_cache_dim,
    kv_projection_dim,
    output_loss_factor,
    query_projection_dim,
    shared_expert_intermediate_total,
)
from hcu_train_simulator.modeling.cost_registry import (
    get_activation_cost_handler,
    get_parameter_cost_handler,
)
from hcu_train_simulator.modeling.spec import (
    BiasDropoutAdd,
    Embedding,
    GatedSelfAttention,
    MLP,
    MLASelfAttention,
    MoELayer,
    ModuleSpec,
    SelfAttention,
    SharedExpertMLP,
    TEGroupedMLP,
    TESequentialMLP,
    TENorm,
    TopKRouter,
    TransformerLayer,
    VisionTransformerLayer,
    module_is,
    module_type_name,
)


@dataclass(frozen=True)
class ParameterEstimate:
    total_elements: float = 0.0
    expert_elements: float = 0.0

    @property
    def dense_elements(self):
        return self.total_elements - self.expert_elements

    def __add__(self, other):
        return ParameterEstimate(
            self.total_elements + other.total_elements,
            self.expert_elements + other.expert_elements,
        )


@dataclass(frozen=True)
class ActivationEstimate:
    saved_elements: float = 0.0
    workspace_elements: float = 0.0

    @property
    def total_elements(self):
        return self.saved_elements + self.workspace_elements

    def __add__(self, other):
        return ActivationEstimate(
            self.saved_elements + other.saved_elements,
            max(self.workspace_elements, other.workspace_elements),
        )


def _child_specs(spec):
    submodules = spec.submodules
    if submodules is None:
        return
    for name, value in vars(submodules).items():
        if isinstance(value, ModuleSpec):
            yield name, value
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, ModuleSpec):
                    yield f"{name}[{index}]", item


class ModuleSpecMemoryModel:
    """Estimate parameters and retained activations from a ModuleSpec tree."""

    def __init__(self, config):
        self.config = config
        for key, value in flatten_dataclass(config).items():
            setattr(self, key, value)
        self.model_spec = config.model_spec

    def _scope_width(self, scope):
        widths = {
            "hidden_size": self.hidden_size,
            "head_dim": self.kv_channels,
            "q_lora_rank": self.q_lora_rank or 0,
            "kv_lora_rank": self.kv_lora_rank or 0,
            "vision_hidden_size": self.config.vision.hidden_size,
        }
        return widths.get(scope, 0)

    def _own_parameters(self, spec, sharded=True):
        role = spec.metainfo.get("role")
        fused_norm = self._scope_width(spec.params.get("fused_norm_scope"))
        tp = self.tp_size if sharded else 1
        ep = self.ep_size if sharded else 1
        etp = self.etp_size if sharded else 1

        registered = get_parameter_cost_handler(spec)
        if registered is not None:
            return registered(self, spec, sharded)

        if module_is(spec, Embedding):
            return ParameterEstimate(self.vocab_size * self.hidden_size / tp)
        if module_is(spec, TENorm):
            width = self._scope_width(spec.params.get("scope", "hidden_size"))
            if spec.params.get("normalization") == "LayerNorm":
                width *= 2
            return ParameterEstimate(width)
        if role == "lm_head":
            elements = 0 if self.tie_word_embeddings else self.vocab_size * self.hidden_size / tp
            return ParameterEstimate(elements)
        if role == "qkv_weight":
            projection = (
                self.num_attention_heads + 2 * self.num_query_groups
            ) * self.kv_channels * self.hidden_size / tp
            return ParameterEstimate(fused_norm + projection)
        if role in {"mla_q_proj", "mla_q_down"}:
            if role == "mla_q_down":
                projection = self.hidden_size * self.q_lora_rank
            else:
                projection = self.hidden_size * query_projection_dim(self) / tp
            return ParameterEstimate(fused_norm + projection)
        if role == "mla_q_up":
            if not self.q_lora_rank:
                return ParameterEstimate()
            return ParameterEstimate(self.q_lora_rank * query_projection_dim(self) / tp)
        if role == "mla_kv_down":
            return ParameterEstimate(self.hidden_size * (self.kv_lora_rank + self.qk_rope_head_dim))
        if role == "mla_kv_up":
            return ParameterEstimate(self.kv_lora_rank * kv_projection_dim(self) / tp)
        if role == "attn_proj":
            return ParameterEstimate(attention_output_dim(self) * self.hidden_size / tp)
        if role == "linear_fc1":
            projection = 2 * self.ffn_hidden_size * self.hidden_size / tp
            return ParameterEstimate(fused_norm + projection)
        if role == "linear_fc2":
            return ParameterEstimate(self.ffn_hidden_size * self.hidden_size / tp)
        if module_is(spec, TopKRouter) or role == "topk_router":
            return ParameterEstimate(self.num_moe_experts * self.hidden_size)
        if spec.module in {TEGroupedMLP, TESequentialMLP} or role == "routed_experts":
            elements = (
                self.num_moe_experts
                / ep
                * self.moe_ffn_hidden_size
                * self.hidden_size
                / etp
                * 3
            )
            return ParameterEstimate(elements, elements)
        if module_is(spec, SharedExpertMLP) or role == "shared_experts":
            elements = shared_expert_intermediate_total(self) * self.hidden_size / tp * 3
            if spec.params.get("gated"):
                elements += self.hidden_size
            return ParameterEstimate(elements)
        return ParameterEstimate()

    def parameter_estimate(self, spec=None, *, sharded=True):
        spec = spec or self.model_spec
        result = self._own_parameters(spec, sharded=sharded)
        for _, child in _child_specs(spec):
            result += self.parameter_estimate(child, sharded=sharded)
        return result

    def global_parameter_estimate(self, spec=None):
        return self.parameter_estimate(spec, sharded=False)

    def model_parameter_estimate(self, spec=None):
        """Return global backbone parameters, excluding auxiliary MTP modules."""

        spec = spec or self.model_spec
        if (
            spec.metainfo.get("is_mtp_layer", False)
            or spec.metainfo.get("excluded_from_model_size", False)
        ):
            return ParameterEstimate()
        result = self._own_parameters(spec, sharded=False)
        for _, child in _child_specs(spec):
            result += self.model_parameter_estimate(child)
        return result

    def _tokens(self):
        return self.micro_batch_size * self.seq_length

    def _vision_tokens(self):
        if not self.config.vision.enabled:
            return 0
        seq = self.config.parallel.vision_seq_length or self.config.vision.num_position_embeddings
        return self.micro_batch_size * self.config.parallel.vision_num_images * seq

    def _own_activations(self, spec, layer_uses_moe=False):
        role = spec.metainfo.get("role")
        tokens = self._tokens()

        registered = get_activation_cost_handler(spec)
        if registered is not None:
            return registered(self, spec, layer_uses_moe)

        if module_is(spec, Embedding):
            return ActivationEstimate(tokens * self.hidden_size)
        if spec.module in {SelfAttention, MLASelfAttention, GatedSelfAttention}:
            if not layer_uses_moe:
                # Preserve the established Megatron dense-layer approximation:
                # attention (16H) + two BDA nodes (2H) + MLP (4F).
                return ActivationEstimate(tokens * self.hidden_size * 16 / self.tp_size)
            core = tokens * kv_cache_dim(self) / self.tp_size * 2
            if self.cp_size > 1:
                core *= 2
            projection = tokens * self.hidden_size / self.tp_size * 8
            return ActivationEstimate(core + projection)
        if module_is(spec, MLP):
            return ActivationEstimate(tokens * self.ffn_hidden_size / self.tp_size * 4)
        if module_is(spec, MoELayer):
            pre_dispatch_tokens = self.micro_batch_size * (
                self.seq_length / self.tp_size * self.etp_size
            )
            router = pre_dispatch_tokens * self.hidden_size * 2
            dispatch = pre_dispatch_tokens * self.hidden_size * self.moe_router_topk
            expert_base = (
                self.micro_batch_size
                * (self.seq_length / self.tp_size * self.etp_size * self.moe_router_topk)
                * self.moe_ffn_hidden_size
                / self.etp_size
            )
            experts = expert_base * 3
            shared = tokens * shared_expert_intermediate_total(self) / self.tp_size * 3
            return ActivationEstimate(router + dispatch + experts + shared)
        if module_is(spec, BiasDropoutAdd) and role != "vision_residual_add":
            return ActivationEstimate(tokens * self.hidden_size / self.tp_size)
        if module_is(spec, TENorm) and role == "mlp_rmsnorm":
            return ActivationEstimate(tokens * self.hidden_size / self.tp_size)
        if module_is(spec, TENorm) and role == "final_rmsnorm":
            return ActivationEstimate(tokens * self.hidden_size)
        if role == "lm_head":
            elements = (
                tokens
                * self.vocab_size
                / self.tp_size
                * output_loss_factor(self)
                * 3
            )
            return ActivationEstimate(elements)
        return ActivationEstimate()

    def activation_estimate(self, spec=None, layer_uses_moe=False):
        spec = spec or self.model_spec
        if module_is(spec, TransformerLayer):
            layer_uses_moe = module_is(spec.submodules.mlp, MoELayer)
        if module_is(spec, VisionTransformerLayer):
            layer_uses_moe = False

        result = self._own_activations(spec, layer_uses_moe=layer_uses_moe)
        for _, child in _child_specs(spec):
            result += self.activation_estimate(child, layer_uses_moe=layer_uses_moe)
        return result

    def memory_rows(self, spec=None, path="model", layer_uses_moe=False):
        """Return non-zero intrinsic costs for every concrete ModuleSpec node."""

        spec = spec or self.model_spec
        if module_is(spec, TransformerLayer):
            layer_uses_moe = module_is(spec.submodules.mlp, MoELayer)
        params = self._own_parameters(spec)
        activations = self._own_activations(spec, layer_uses_moe=layer_uses_moe)
        rows = []
        if params.total_elements or activations.total_elements:
            rows.append(
                {
                    "module_path": path,
                    "module_type": module_type_name(spec),
                    "role": spec.metainfo.get("role") or "/",
                    "param_elems": int(params.total_elements),
                    "expert_param_elems": int(params.expert_elements),
                    "saved_act_elems": int(activations.saved_elements),
                    "workspace_elems": int(activations.workspace_elements),
                }
            )
        for name, child in _child_specs(spec):
            rows.extend(
                self.memory_rows(
                    child,
                    f"{path}.{name}",
                    layer_uses_moe=layer_uses_moe,
                )
            )
        return rows
