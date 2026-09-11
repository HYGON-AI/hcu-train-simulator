# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from hcu_train_simulator.config.utils import flatten_dataclass
from hcu_train_simulator.context import get_config
from hcu_train_simulator.modeling.architecture import (
    attention_output_dim,
    attention_score_dim,
    kv_projection_dim,
    output_loss_factor,
    query_projection_dim,
    shared_expert_intermediate_total,
    has_qk_norm,
    uses_mla,
    uses_moe,
    value_head_dim,
)
from hcu_train_simulator.modeling.parameters import ParameterModel
from hcu_train_simulator.modeling.compute_registry import (
    get_attention_lowering,
    get_component_lowering,
    get_first_gemm_part,
    resolve_gemm_shape,
)
from hcu_train_simulator.modeling.spec import (
    IdentityOp,
    MLP,
    MLASelfAttention,
    MoELayer,
    SelfAttention,
    module_is,
    module_type_name,
)
from hcu_train_simulator.parallelism import (
    build_pipeline_layer_layout,
    build_pp_layer_counts,
    get_num_microbatches,
)


NON_GEMM_OPS = {
    "rmsnorm": (7, 12),
    "layernorm": (8, 13),
    "qk_norm": (7, 12),
    "rope": (8, 8),
    "attention_softmax": (5, 8),
    "residual_dropout_add": (3, 3),
    "swiglu": (8, 12),
    "router_select": (6, 4),
    "cross_entropy": (6, 4),
    "optimizer_adamw": (14, 0),
    "sigmoid_gate": (5, 8),
    "causal_conv1d": (6, 10),
    "gated_delta_rule": (12, 24),
    "l2norm": (6, 10),
    "gated_rmsnorm": (10, 16),
    "gelu": (8, 12),
    "position_add": (1, 1),
    "dsa_score_reduce": (3, 5),
    # Comparison/select work is intentionally modeled as a non-GEMM kernel.
    # The coefficient is a planning approximation for radix/partial top-k.
    "dsa_topk": (8, 0),
}


class ComputeModel:
    def __init__(self):
        config = get_config()
        self.config = config

        flat = flatten_dataclass(config)
        for k, v in flat.items():
            setattr(self, k, v)
        self.model_spec = config.model_spec

    def calc_compute_time(self, m, n, k, num_compute):
        return 2 * m * n * k / (self.fp16_tflops * self.gemm_efficiency) / 1e9 * num_compute

    def calc_non_gemm_time(self, elements, ops_per_element, num_compute=1, efficiency=None):
        if not elements or not ops_per_element or not num_compute:
            return 0.0
        eff = efficiency if efficiency is not None else getattr(self, "non_gemm_efficiency", 0.08)
        eff = max(float(eff or 0.08), 1e-9)
        return elements * ops_per_element / (self.fp16_tflops * eff) / 1e9 * num_compute

    def local_seq_length(self):
        """Return the sequence shard processed by one context-parallel rank."""

        return self.seq_length // max(1, self.cp_size)

    def token_count(self):
        """Return the token count processed by one rank for one microbatch."""

        return self.micro_batch_size * self.local_seq_length()

    def moe_tokens_per_local_expert(self):
        sequence_shard = self.tp_size if self.sequence_parallel else 1
        dispatched_tokens_per_rank = (
            self.token_count()
            * self.moe_router_topk
            * self.ep_size
            * self.etp_size
            / sequence_shard
        )
        return max(1, dispatched_tokens_per_rank / self.num_moe_experts)

    def transformer_layers_on_critical_rank(self):
        return self.num_layers // self.pp_size + (self.mtp_num_layers or 0)

    def transformer_compute_count(self):
        return self.microbatch_compute_count() * self.transformer_layers_on_critical_rank()

    def microbatch_compute_count(self):
        return get_num_microbatches(
            self.global_batch_size,
            self.micro_batch_size,
            self.dp_size,
        )

    def output_compute_count(self):
        return self.microbatch_compute_count() * output_loss_factor(self)

    def lm_head(self):
        num_compute = self.output_compute_count()
        return num_compute, self.vocab_size // self.tp_size, self.hidden_size, self.token_count()

    def qkv_weight(self):
        num_compute = self.transformer_compute_count()
        m = (
            self.num_attention_heads * self.kv_channels // self.tp_size
            + self.num_query_groups * self.kv_channels // self.tp_size * 2
        )
        return num_compute, m, self.hidden_size, self.token_count()

    def mla_q_down(self):
        num_compute = self.transformer_compute_count()
        rank = self.q_lora_rank or query_projection_dim(self) // self.tp_size
        return num_compute, rank, self.hidden_size, self.token_count()

    def mla_q_up(self):
        num_compute = self.transformer_compute_count()
        k = self.q_lora_rank or self.hidden_size
        return num_compute, query_projection_dim(self) // self.tp_size, k, self.token_count()

    def mla_kv_down(self):
        num_compute = self.transformer_compute_count()
        return num_compute, self.kv_lora_rank + self.qk_rope_head_dim, self.hidden_size, self.token_count()

    def mla_kv_up(self):
        num_compute = self.transformer_compute_count()
        return num_compute, kv_projection_dim(self) // self.tp_size, self.kv_lora_rank, self.token_count()

    def attn_proj(self):
        num_compute = self.transformer_compute_count()
        return num_compute, self.hidden_size, attention_output_dim(self) // self.tp_size, self.token_count()

    def shared_expert_gate(self):
        return self.transformer_compute_count(), 1, self.hidden_size, self.token_count()

    def linear_fc1(self):
        num_compute = self.transformer_compute_count()
        return num_compute, self.ffn_hidden_size // self.tp_size * 2, self.hidden_size, self.token_count()

    def linear_fc2(self):
        num_compute = self.transformer_compute_count()
        return num_compute, self.hidden_size, self.ffn_hidden_size // self.tp_size, self.token_count()

    def moe_linear_fc1(self):
        local_experts = self.num_moe_experts // self.ep_size
        tokens_per_expert = self.moe_tokens_per_local_expert()
        num_compute = self.transformer_compute_count() * local_experts
        moe_intermediate_rank = self.moe_ffn_hidden_size // max(1, self.etp_size)
        return num_compute, moe_intermediate_rank * 2, tokens_per_expert, self.hidden_size

    def moe_linear_fc2(self):
        local_experts = self.num_moe_experts // self.ep_size
        tokens_per_expert = self.moe_tokens_per_local_expert()
        num_compute = self.transformer_compute_count() * local_experts
        moe_intermediate_rank = self.moe_ffn_hidden_size // max(1, self.etp_size)
        return num_compute, self.hidden_size, tokens_per_expert, moe_intermediate_rank

    def shared_expert_fc1(self):
        num_compute = self.transformer_compute_count()
        shared_intermediate = shared_expert_intermediate_total(self)
        return num_compute, shared_intermediate // self.tp_size * 2, self.hidden_size, self.token_count()

    def shared_expert_fc2(self):
        num_compute = self.transformer_compute_count()
        shared_intermediate = shared_expert_intermediate_total(self)
        return num_compute, self.hidden_size, shared_intermediate // self.tp_size, self.token_count()

    def topk_router(self):
        num_compute = self.transformer_compute_count()
        return num_compute, self.num_moe_experts, self.hidden_size, self.token_count() // self.tp_size

    def transformer_element_count(self, width=None, tp_sharded=False):
        shard = self.tp_size if tp_sharded else 1
        return self.token_count() * (width or self.hidden_size) / shard

    def qk_norm_element_count(self):
        if not has_qk_norm(self):
            return 0
        return self.transformer_element_count(
            (self.num_attention_heads + self.num_query_groups) * self.kv_channels,
            tp_sharded=True,
        )

    def rope_element_count(self):
        if uses_mla(self):
            rope_heads = self.num_attention_heads / self.tp_size + 1
            return self.token_count() * rope_heads * self.qk_rope_head_dim
        rope_heads = (self.num_attention_heads + self.num_query_groups) / self.tp_size
        return self.token_count() * rope_heads * self.kv_channels * self.partial_rotary_factor

    def attention_softmax_element_count(self):
        q_heads = self.num_attention_heads / self.tp_size
        # A CP rank owns local queries while attending to the global KV context.
        return self.token_count() * q_heads * self.seq_length

    def dense_swiglu_element_count(self):
        return self.transformer_element_count(self.ffn_hidden_size, tp_sharded=True)

    def moe_swiglu_element_count(self):
        tokens_per_expert = self.moe_tokens_per_local_expert()
        return tokens_per_expert * (self.moe_ffn_hidden_size // max(1, self.etp_size))

    def shared_swiglu_element_count(self):
        shared_intermediate = shared_expert_intermediate_total(self)
        if not shared_intermediate:
            return 0
        return self.transformer_element_count(shared_intermediate, tp_sharded=True)

    def router_select_element_count(self):
        if not uses_moe(self):
            return 0
        return self.token_count() * self.num_moe_experts / self.tp_size

    def cross_entropy_element_count(self):
        return self.token_count() * self.vocab_size / self.tp_size

    def pp_layer_counts(self):
        if self.num_layers_per_vp_stage:
            counts = [self.num_layers_per_vp_stage * max(self.vp_size, 1)] * self.pp_size
        else:
            first_layers = self.decoder_first_pipeline_num_layers or 0
            last_layers = self.decoder_last_pipeline_num_layers or 0
            counts = build_pp_layer_counts(
                self.num_layers,
                self.pp_size,
                first_layers,
                last_layers,
            )
        if self.mtp_num_layers:
            counts[-1] += self.mtp_num_layers
        return counts

    def optimizer_param_elements(self):
        params = ParameterModel()
        layout = build_pipeline_layer_layout(self.model_spec, get_config().parallel)
        max_params = 0
        for pp_rank, virtual_stages in enumerate(layout):
            local_params = sum(
                params.estimate(layer).total_elements
                for layer_specs in virtual_stages
                for layer in layer_specs
            )
            if pp_rank == 0:
                local_params += params.input_embed()
                vision_spec = getattr(self.model_spec.submodules, "vision_model", None)
                if vision_spec is not None:
                    local_params += params.estimate(vision_spec).total_elements
            if pp_rank == self.pp_size - 1:
                mtp = params.mtp()
                if mtp is not None:
                    local_params += mtp.total_elements
                local_params += params.output_layer()
            max_params = max(max_params, local_params)

        if self.use_distributed_optimizer:
            return max_params / max(1, self.dp_size * self.cp_size)
        return max_params

    def fa_compute(self):
        num_compute = self.transformer_compute_count()
        per_head_qk = attention_score_dim(self)
        per_head_v = value_head_dim(self)
        local_seq = self.local_seq_length()
        forward_flops = (
            2
            * self.micro_batch_size
            * local_seq
            * self.seq_length
            * self.num_attention_heads
            / self.tp_size
            * (per_head_qk + per_head_v)
        )
        backward_flops = (
            5
            * self.micro_batch_size
            * local_seq
            * self.seq_length
            * self.num_attention_heads
            / self.tp_size
            * (per_head_qk + per_head_v)
        )

        forward_time = forward_flops * num_compute / (self.fp16_tflops * self.gemm_efficiency) / 1e9
        backward_time = backward_flops * num_compute / (self.fp16_tflops * self.gemm_efficiency * 0.5) / 1e9
        return num_compute, forward_time, backward_time

    def attention_parts(self):
        if uses_mla(self):
            return ["mla_q_down", "mla_q_up", "mla_kv_down", "mla_kv_up", "attn_proj"]
        return ["qkv_weight", "attn_proj"]

    def add_gemm_part(self, result, part, *, count_scale=1.0, module_spec=None, module_path=None):
        num_compute, m, k, n = resolve_gemm_shape(self, part)
        num_compute *= count_scale
        cur_part_time = self.calc_compute_time(m, n, k, num_compute)
        row = {
                "model_part": part,
                "b": self.micro_batch_size,
                "m": m,
                "n": n,
                "k": k,
                "shape": f"M={m}, N={n}, K={k}",
                "compute_count": num_compute,
                "forward_ms": cur_part_time,
                "backward_ms": cur_part_time * 2,
                "op_type": "gemm",
            }
        if part in {"moe_linear_fc1", "moe_linear_fc2"}:
            row["num_gemms"] = self.num_moe_experts // self.ep_size
            row["grouped_gemm"] = bool(self.moe_grouped_gemm)
            row["tokens_per_expert"] = int(k)
        self._attach_spec_metadata(row, module_spec, module_path)
        result.append(row)

    def add_non_gemm_part(
        self,
        result,
        part,
        elements,
        compute_count,
        ops_key,
        *,
        covered_by_measured=None,
        efficiency=None,
        shape=None,
        module_spec=None,
        module_path=None,
    ):
        if not elements or not compute_count:
            return
        forward_ops, backward_ops = NON_GEMM_OPS[ops_key]
        forward_ms = self.calc_non_gemm_time(
            elements,
            forward_ops,
            compute_count,
            efficiency=efficiency,
        )
        backward_ms = self.calc_non_gemm_time(
            elements,
            backward_ops,
            compute_count,
            efficiency=efficiency,
        )
        row = {
            "model_part": part,
            "b": self.micro_batch_size,
            "m": "/",
            "n": "/",
            "k": "/",
            "shape": shape or f"elements={int(elements)}",
            "elements": int(elements),
            "total_elements": int(elements * compute_count),
            "compute_count": round(compute_count, 6) if isinstance(compute_count, float) else compute_count,
            "forward_ms": forward_ms,
            "backward_ms": backward_ms,
            "op_type": "non_gemm",
            "forward_ops_per_element": forward_ops,
            "backward_ops_per_element": backward_ops,
            "ops_per_element": forward_ops + backward_ops,
        }
        if covered_by_measured:
            row["covered_by_measured"] = covered_by_measured
        self._attach_spec_metadata(row, module_spec, module_path)
        result.append(row)

    def _attach_spec_metadata(self, row, module_spec, module_path):
        if module_spec is not None:
            row["module_type"] = module_type_name(module_spec)
            row["module_role"] = module_spec.metainfo.get("role") or "/"
        if module_path:
            row["module_path"] = module_path

    def add_flash_attention_after(
        self,
        result,
        part,
        *,
        count_scale=1.0,
        mla=None,
        module_spec=None,
        module_path=None,
    ):
        mla = uses_mla(self) if mla is None else mla
        trigger = "mla_kv_up" if mla else "qkv_weight"
        if part != trigger:
            return
        fa_compute, fa_fwd, fa_bwd = self.fa_compute()
        fa_compute *= count_scale
        local_seq = self.local_seq_length()
        sequence_shape = f"batch={self.micro_batch_size}, seq={local_seq}"
        if self.cp_size > 1:
            sequence_shape += f", kv_seq={self.seq_length}"
        row = {
                "model_part": "flash_attn",
                "b": self.micro_batch_size,
                "m": "/",
                "n": "/",
                "k": "/",
                "shape": (
                    f"{sequence_shape}, "
                    f"heads={self.num_attention_heads // self.tp_size}, head_dim={attention_score_dim(self)}"
                ),
                "compute_count": fa_compute,
                "seq": local_seq,
                "kv_seq": self.seq_length,
                "heads": self.num_attention_heads // self.tp_size,
                "head_dim": attention_score_dim(self),
                "value_head_dim": value_head_dim(self),
                "causal": True,
                "forward_ms": fa_fwd * count_scale,
                "backward_ms": fa_bwd * count_scale,
                "op_type": "flash_attention",
            }
        self._attach_spec_metadata(row, module_spec, module_path)
        result.append(row)

    def add_common_attention_non_gemm(
        self,
        result,
        compute_count=None,
        *,
        attention_spec=None,
        module_path=None,
    ):
        compute_count = compute_count or self.transformer_compute_count()
        has_layer_qk_norm = has_qk_norm(self)
        if attention_spec is not None and module_is(attention_spec, SelfAttention):
            submodules = attention_spec.submodules
            has_layer_qk_norm = any(
                norm is not None and not module_is(norm, IdentityOp)
                for norm in (submodules.q_layernorm, submodules.k_layernorm)
            )
        self.add_non_gemm_part(
            result,
            "qk_norm",
            self.qk_norm_element_count() if has_layer_qk_norm else 0,
            compute_count,
            "qk_norm",
            module_spec=attention_spec,
            module_path=f"{module_path}.qk_norm" if module_path else None,
        )
        self.add_non_gemm_part(
            result,
            "rope",
            self.rope_element_count(),
            compute_count,
            "rope",
            module_spec=attention_spec,
            module_path=f"{module_path}.rope" if module_path else None,
        )
        self.add_non_gemm_part(
            result,
            "attention_softmax",
            self.attention_softmax_element_count(),
            compute_count,
            "attention_softmax",
            covered_by_measured="flash_attn",
            module_spec=(attention_spec.submodules.core_attention if attention_spec else None),
            module_path=f"{module_path}.core_attention" if module_path else None,
        )

    def add_output_non_gemm(self, result, include_optimizer=True):
        final_norm_spec = self.model_spec.submodules.decoder.submodules.layer_norm
        output_spec = self.model_spec.submodules.output_layer
        self.add_non_gemm_part(
            result,
            "final_rmsnorm",
            self.transformer_element_count(),
            self.microbatch_compute_count(),
            "rmsnorm",
            module_spec=final_norm_spec,
            module_path="model.decoder.final_norm",
        )
        self.add_non_gemm_part(
            result,
            "cross_entropy",
            self.cross_entropy_element_count(),
            self.output_compute_count(),
            "cross_entropy",
            module_spec=output_spec,
            module_path="model.output_layer.cross_entropy",
        )
        if not include_optimizer:
            return
        benchmark_numel = max(1, int(getattr(self, "benchmark_optimizer_numel", 16_777_216) or 16_777_216))
        optimizer_count = self.optimizer_param_elements() / benchmark_numel
        self.add_non_gemm_part(
            result,
            "optimizer_step",
            benchmark_numel,
            optimizer_count,
            "optimizer_adamw",
            efficiency=getattr(self, "optimizer_efficiency", 0.04),
            shape=f"params_per_chunk={benchmark_numel}",
            module_spec=self.model_spec,
            module_path="model.optimizer",
        )

    def _add_transformer_layer(self, result, layer_spec, module_path, compute_count):
        submodules = layer_spec.submodules
        attention = submodules.self_attention
        mlp = submodules.mlp
        count_scale = compute_count / max(self.transformer_compute_count(), 1e-12)

        self.add_non_gemm_part(
            result,
            "attention_rmsnorm",
            self.transformer_element_count(),
            compute_count,
            "rmsnorm",
            covered_by_measured=(
                get_first_gemm_part(attention)
                or ("qkv_weight" if module_is(attention, SelfAttention) else None)
            ),
            module_spec=submodules.input_layernorm,
            module_path=f"{module_path}.input_layernorm",
        )

        attention_lowering = get_attention_lowering(attention)
        if attention_lowering is not None:
            attention_lowering(
                self,
                result,
                attention,
                module_path,
                compute_count,
                count_scale,
            )
        elif module_is(attention, SelfAttention):
            attention_parts = [
                ("qkv_weight", attention.submodules.linear_qkv, "linear_qkv"),
                ("attn_proj", attention.submodules.linear_proj, "linear_proj"),
            ]
            first_part, first_spec, first_name = attention_parts[0]
            self.add_gemm_part(
                result,
                first_part,
                count_scale=count_scale,
                module_spec=first_spec,
                module_path=f"{module_path}.self_attention.{first_name}",
            )
            self.add_flash_attention_after(
                result,
                first_part,
                count_scale=count_scale,
                mla=False,
                module_spec=attention.submodules.core_attention,
                module_path=f"{module_path}.self_attention.core_attention",
            )
            self.add_common_attention_non_gemm(
                result,
                compute_count,
                attention_spec=attention,
                module_path=f"{module_path}.self_attention",
            )
            part, part_spec, part_name = attention_parts[1]
            self.add_gemm_part(
                result,
                part,
                count_scale=count_scale,
                module_spec=part_spec,
                module_path=f"{module_path}.self_attention.{part_name}",
            )
        elif module_is(attention, MLASelfAttention):
            mla_submodules = attention.submodules
            if mla_submodules.linear_q_down_proj is not None:
                q_parts = [
                    ("mla_q_down", mla_submodules.linear_q_down_proj, "linear_q_down_proj"),
                    ("mla_q_up", mla_submodules.linear_q_up_proj, "linear_q_up_proj"),
                ]
            else:
                # Without Q-LoRA, linear_q_proj is the direct hidden->Q
                # projection.  The existing mla_q_up shape already represents
                # that GEMM when q_lora_rank is absent.
                q_parts = [("mla_q_up", mla_submodules.linear_q_proj, "linear_q_proj")]
            mla_parts = q_parts + [
                ("mla_kv_down", mla_submodules.linear_kv_down_proj, "linear_kv_down_proj"),
                ("mla_kv_up", mla_submodules.linear_kv_up_proj, "linear_kv_up_proj"),
            ]
            for part, part_spec, part_name in mla_parts:
                self.add_gemm_part(
                    result,
                    part,
                    count_scale=count_scale,
                    module_spec=part_spec,
                    module_path=f"{module_path}.self_attention.{part_name}",
                )
                self.add_flash_attention_after(
                    result,
                    part,
                    count_scale=count_scale,
                    mla=True,
                    module_spec=mla_submodules.core_attention,
                    module_path=f"{module_path}.self_attention.core_attention",
                )
            self.add_common_attention_non_gemm(
                result,
                compute_count,
                attention_spec=attention,
                module_path=f"{module_path}.self_attention",
            )
            self.add_gemm_part(
                result,
                "attn_proj",
                count_scale=count_scale,
                module_spec=mla_submodules.linear_proj,
                module_path=f"{module_path}.self_attention.linear_proj",
            )
        else:
            raise NotImplementedError(f"no compute estimator for attention ModuleSpec {attention.module!r}")

        self.add_non_gemm_part(
            result,
            "mlp_rmsnorm",
            self.transformer_element_count(),
            compute_count,
            "rmsnorm",
            covered_by_measured="linear_fc1" if module_is(mlp, MLP) else None,
            module_spec=submodules.pre_mlp_layernorm,
            module_path=f"{module_path}.pre_mlp_layernorm",
        )
        if module_is(mlp, MLP):
            self.add_gemm_part(
                result,
                "linear_fc1",
                count_scale=count_scale,
                module_spec=mlp.submodules.linear_fc1,
                module_path=f"{module_path}.mlp.linear_fc1",
            )
            self.add_non_gemm_part(
                result,
                "swiglu_activation",
                self.dense_swiglu_element_count(),
                compute_count,
                "swiglu",
                module_spec=mlp.submodules.activation_func,
                module_path=f"{module_path}.mlp.activation_func",
            )
            self.add_gemm_part(
                result,
                "linear_fc2",
                count_scale=count_scale,
                module_spec=mlp.submodules.linear_fc2,
                module_path=f"{module_path}.mlp.linear_fc2",
            )
        elif module_is(mlp, MoELayer):
            moe_submodules = mlp.submodules
            self.add_gemm_part(
                result,
                "topk_router",
                count_scale=count_scale,
                module_spec=moe_submodules.router,
                module_path=f"{module_path}.mlp.router",
            )
            self.add_non_gemm_part(
                result,
                "router_select",
                self.router_select_element_count(),
                compute_count,
                "router_select",
                covered_by_measured="topk_router",
                module_spec=moe_submodules.router,
                module_path=f"{module_path}.mlp.router",
            )
            self.add_gemm_part(
                result,
                "moe_linear_fc1",
                count_scale=count_scale,
                module_spec=moe_submodules.experts,
                module_path=f"{module_path}.mlp.experts.linear_fc1",
            )
            self.add_non_gemm_part(
                result,
                "moe_swiglu_activation",
                self.moe_swiglu_element_count(),
                compute_count * (self.num_moe_experts // self.ep_size),
                "swiglu",
                module_spec=moe_submodules.experts,
                module_path=f"{module_path}.mlp.experts.activation_func",
            )
            self.add_gemm_part(
                result,
                "moe_linear_fc2",
                count_scale=count_scale,
                module_spec=moe_submodules.experts,
                module_path=f"{module_path}.mlp.experts.linear_fc2",
            )
            if moe_submodules.shared_experts is not None:
                if moe_submodules.shared_experts.params.get("gated"):
                    self.add_gemm_part(
                        result,
                        "shared_expert_gate",
                        count_scale=count_scale,
                        module_spec=moe_submodules.shared_experts,
                        module_path=f"{module_path}.mlp.shared_experts.gate",
                    )
                self.add_gemm_part(
                    result,
                    "shared_expert_fc1",
                    count_scale=count_scale,
                    module_spec=moe_submodules.shared_experts,
                    module_path=f"{module_path}.mlp.shared_experts.linear_fc1",
                )
                self.add_non_gemm_part(
                    result,
                    "shared_swiglu_activation",
                    self.shared_swiglu_element_count(),
                    compute_count,
                    "swiglu",
                    module_spec=moe_submodules.shared_experts,
                    module_path=f"{module_path}.mlp.shared_experts.activation_func",
                )
                self.add_gemm_part(
                    result,
                    "shared_expert_fc2",
                    count_scale=count_scale,
                    module_spec=moe_submodules.shared_experts,
                    module_path=f"{module_path}.mlp.shared_experts.linear_fc2",
                )
        else:
            raise NotImplementedError(f"no compute estimator for MLP ModuleSpec {mlp.module!r}")

        for bda_name in ("self_attn_bda", "mlp_bda"):
            bda_spec = getattr(submodules, bda_name)
            self.add_non_gemm_part(
                result,
                "residual_dropout_add",
                self.transformer_element_count(tp_sharded=True),
                compute_count,
                "residual_dropout_add",
                module_spec=bda_spec,
                module_path=f"{module_path}.{bda_name}",
            )

    def _merge_module_rows(self, rows):
        merged = []
        index = {}
        for row in rows:
            key = (
                row.get("model_part"),
                row.get("op_type"),
                row.get("shape"),
                row.get("covered_by_measured"),
            )
            if key not in index:
                copy = dict(row)
                copy["module_count"] = 1
                copy["_first_module_path"] = copy.get("module_path")
                merged.append(copy)
                index[key] = copy
                continue
            target = index[key]
            for field in ("compute_count", "forward_ms", "backward_ms", "total_elements"):
                if field in row:
                    target[field] = target.get(field, 0) + row[field]
            target["module_count"] += 1
        for row in merged:
            first_path = row.pop("_first_module_path", None)
            if first_path and row["module_count"] > 1:
                row["module_path"] = f"{first_path} [x{row['module_count']}]"
            row["forward_ms"] = round(row["forward_ms"], 4)
            row["backward_ms"] = round(row["backward_ms"], 4)
            if isinstance(row.get("compute_count"), float):
                row["compute_count"] = round(row["compute_count"], 6)
        return merged

    def spec_summary(self):
        layout = build_pipeline_layer_layout(self.model_spec, get_config().parallel)
        num_microbatches = get_num_microbatches(
            self.global_batch_size,
            self.micro_batch_size,
            self.dp_size,
        )
        stage_rows = []
        for pp_rank, virtual_stages in enumerate(layout):
            rows = []
            vision_spec = getattr(self.model_spec.submodules, "vision_model", None)
            if pp_rank == 0 and vision_spec is not None:
                component_lowering = get_component_lowering(vision_spec)
                if component_lowering is None:
                    raise NotImplementedError(
                        f"no compute lowering registered for model component {vision_spec.module!r}"
                    )
                component_lowering(self, rows, vision_spec, "model.vision_model")
            for vp_rank, layer_specs in enumerate(virtual_stages):
                for layer_index, layer_spec in enumerate(layer_specs):
                    self._add_transformer_layer(
                        rows,
                        layer_spec,
                        f"model.decoder.pp[{pp_rank}].vp[{vp_rank}].layers[{layer_index}]",
                        num_microbatches,
                    )
            if pp_rank == self.pp_size - 1:
                mtp_spec = getattr(self.model_spec.submodules, "mtp", None)
                if mtp_spec is not None:
                    component_lowering = get_component_lowering(mtp_spec)
                    if component_lowering is None:
                        raise NotImplementedError(
                            f"no compute lowering registered for model component {mtp_spec.module!r}"
                        )
                    component_lowering(self, rows, mtp_spec, "model.mtp")
                output_spec = self.model_spec.submodules.output_layer
                self.add_gemm_part(
                    rows,
                    "lm_head",
                    module_spec=output_spec,
                    module_path="model.output_layer",
                )
                self.add_output_non_gemm(rows, include_optimizer=False)
            stage_rows.append(rows)

        critical_rank, result = max(
            enumerate(stage_rows),
            key=lambda item: sum(row["forward_ms"] + row["backward_ms"] for row in item[1]),
        )
        self.critical_pp_rank = critical_rank
        benchmark_numel = max(1, int(getattr(self, "benchmark_optimizer_numel", 16_777_216) or 16_777_216))
        optimizer_count = self.optimizer_param_elements() / benchmark_numel
        self.add_non_gemm_part(
            result,
            "optimizer_step",
            benchmark_numel,
            optimizer_count,
            "optimizer_adamw",
            efficiency=getattr(self, "optimizer_efficiency", 0.04),
            shape=f"params_per_chunk={benchmark_numel}",
            module_spec=self.model_spec,
            module_path="model.optimizer",
        )
        self.module_rows = list(result)
        return self._merge_module_rows(result)

    def dense_summary(self):
        return self.spec_summary()

    def moe_summary(self):
        return self.spec_summary()
