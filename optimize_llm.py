######################################################################
# Construct the model architecture
# --------------------------------
# We will use a pre-trained TinyLlama model from Hugging Face. However, usually we only load the
# pre-trained weight from Hugging Face but not the model architecture. We need to construct the
# model architecture by ourselves. Apache TVM prepares a PyTorch-liked API to construct the model
# architecture. We can use the API to construct the model architecture.


import dataclasses
import enum
import os
from pathlib import Path
from pprint import pprint
from typing import List, Optional

import tvm
from tvm import dlight, relax, te, tir
from tvm.relax import register_pipeline
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.relax.frontend.nn.llm.kv_cache import PagedKVCache, TIRPagedKVCache
from tvm.runtime import ShapeTuple


@dataclasses.dataclass
class QwenConfig:
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_attention_heads: int = 14
    num_hidden_layers: int = 24
    rms_norm_eps: float = 1e-06
    vocab_size: int = 151936
    rope_theta: float = 1000000.0
    context_window_size: int = 32768
    prefill_chunk_size: int = 2048
    num_key_value_heads: int = 2
    head_dim: int = 64  # # hidden_size // num_attention_heads


dev = tvm.device("cuda", 0)
target = tvm.target.Target.from_device(dev)


######################################################################
# Next, we define the RoPE mode of the Paged KV cache. The RoPE mode is used to apply the
# Relative Positional Encoding (RoPE) to the query and key tensors. The RoPE mode can be set to
# `NONE`, `NORMAL`, or `INLINE`. If the RoPE mode is `NONE`, the KV cache will not apply RoPE to
# the query and key tensors. If the RoPE mode is `NORMAL`, RoPE will be applied to the key tensor
# before adding the key tensor to the cache. If the RoPE mode is `INLINE`, RoPE will be applied to
# the query and key tensors in the attention kernel on-the-fly.


class RopeMode(enum.IntEnum):
    """The RoPE mode of the Paged KV cache.
    If it is none, the KV cache will not apply RoPE to q and k.
    If it is normal, RoPE will be applied to k before adding k to cache.
    Otherwise, RoPE will be applied to q/k in attention kernel on-the-fly.
    """

    NONE = 0
    NORMAL = 1
    INLINE = 2


######################################################################
# Secondly, we define the model architecture. The model architecture consists of three parts:
#
# - Embedding layer: The embedding layer converts the input token IDs to the hidden states.
# - Decoder layers: The decoder layers are the core of the model. Each decoder layer consists of
#   a self-attention layer and a feed-forward network (FFN) layer.
# - Output layer: The output layer converts the hidden states to the logits.
#
# First we define the FFN layer. Note that the following FFN layer is optimized implementation
# where we fuse the gate and up projection into one kernel.
# The naive implementation of FFN layer is: ``FFN(x) = down_proj(silu(gate(x)) * up(x))``
# We could combine the ``gate`` and ``up`` projection into one kernel for better performance.
# The optimized implementation is:
#
# .. code-block:: python
#
#   concat_x = gate_up(x)
#   gate_x, up_x = split(concat_x, 2, axis=-1)
#   FFN(x) = down_proj(silu(gate_x) * up_x)
#


class QwenFFN(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.gate_up_proj = nn.Linear(
            in_features=config.hidden_size,
            out_features=2 * config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor):
        concat_x1_x2 = self.gate_up_proj(x)
        x1, x2 = op.split(concat_x1_x2, 2, axis=-1)
        return self.down_proj(op.silu(x1) * x2)


######################################################################
# Then we define the self-attention layer. The self-attention layer consists of three parts:
#
# - QKV projection: The QKV projection converts the input hidden states to the query, key, and
#   value tensors.
# - Attention: The attention layer computes the attention scores and applies the softmax
#   operation.
# - Output projection: The output projection converts the attention output to the hidden states.
#
# We perform optimizations on the different parts of the self-attention layer:
#
# - QKV projection: We leverage the horizontal fusion on QKV projection and fuse them into one
#   kernel.
# - Attention: We leverage the horizontal fusion on attention and fuse the QKV projection and


# class QwenAttention(nn.Module):  # re-do this function
#     def __init__(self, config: QwenConfig):
#         self.head_dim = config.head_dim
#         self.num_q_heads = config.num_attention_heads
#         self.num_kv_heads = config.num_key_value_heads
#         # horizontal fusion on QKV projection
#         self.qkv_proj = nn.Linear(
#             in_features=config.hidden_size,
#             out_features=(self.num_q_heads + 2 * self.num_kv_heads) * self.head_dim,
#             bias=False,
#         )
#         self.o_proj = nn.Linear(self.num_q_heads * self.head_dim, config.hidden_size, bias=False)

#     def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
#         d, h_q, h_kv = self.head_dim, self.num_q_heads, self.num_kv_heads
#         b, s, _ = hidden_states.shape
#         # QKV Projection
#         qkv = self.qkv_proj(hidden_states)
#         qkv = op.reshape(qkv, (b, s, h_q + h_kv + h_kv, d))
#         # Attention
#         output = op.reshape(
#             paged_kv_cache.attention_with_fused_qkv(
#                 layer_id, qkv, self.num_q_heads, sm_scale=self.head_dim**-0.5
#             ),
#             (b, s, h_q * d),
#         )
#         # Output Projection
#         return self.o_proj(output)

class QwenAttention(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads

        # Tạo lớp Linear để gộp QKV
        self.qkv_proj = nn.Linear(
            in_features=config.hidden_size,
            out_features=(self.num_q_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_q_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        b, s, _ = hidden_states.shape
        d, h_q, h_kv = self.head_dim, self.num_q_heads, self.num_kv_heads

        # QKV Projection
        qkv = self.qkv_proj(hidden_states)  # (b, s, (h_q + 2 * h_kv) * d)
        qkv = op.reshape(qkv, (b, s, h_q + h_kv + h_kv, d))  # (b, s, num_heads, head_dim)

        print("🔍 qkv.shape trước attention_with_fused_qkv:", qkv.shape)

        # Gọi attention_with_fused_qkv
        attn_output = paged_kv_cache.attention_with_fused_qkv(layer_id, qkv, self.num_q_heads, sm_scale=self.head_dim**-0.5)

        print("🔍 attn_output.shape sau attention_with_fused_qkv:", attn_output.shape)

        # **Reshape lại output đúng kích thước**
        output = op.reshape(attn_output, (b, s, h_q * d))

        # Output projection
        return self.o_proj(output)




######################################################################
# Finally, we define the model architecture with FFN and self-attention layers.


class QwenDecoderLayer(nn.Module):
    def __init__(self, config: QwenConfig):
        rms_norm_eps = config.rms_norm_eps
        self.self_attn = QwenAttention(config)
        self.mlp = QwenFFN(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, -1, rms_norm_eps, bias=False)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, -1, rms_norm_eps, bias=False)

    def forward(self, hidden_states: Tensor, paged_kv_cache: PagedKVCache, layer_id: int):
        hidden_states += self.self_attn(
            self.input_layernorm(hidden_states), paged_kv_cache, layer_id
        )
        hidden_states += self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class QwenModel(nn.Module):
    def __init__(self, config: QwenConfig):
        assert config.hidden_size % config.num_attention_heads == 0
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [QwenDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)

    def forward(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        hidden_states = input_embed
        for layer_id, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, paged_kv_cache, layer_id)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class QwenForCasualLM(nn.Module):
    def __init__(self, config: QwenConfig):
        self.model = QwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.rope_theta = config.rope_theta
        self.dtype = "float32"

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def embed(self, input_ids: Tensor):
        return self.model.embed_tokens(input_ids)

    def get_logits(self, hidden_states: Tensor):
        logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits

    def prefill(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        def _index(x: te.Tensor):  # x[:-1,:]
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states = self.model(input_embed, paged_kv_cache)
        hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        logits = self.get_logits(hidden_states)
        return logits, paged_kv_cache

    def decode(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        hidden_states = self.model(input_embed, paged_kv_cache)
        logits = self.get_logits(hidden_states)
        return logits, paged_kv_cache

    def create_tir_paged_kv_cache(
        self,
        max_batch_size: tir.Var,
        max_total_seq_len: tir.Var,
        prefill_chunk_size: tir.Var,
        page_size: tir.Var,
    ) -> PagedKVCache:
        return TIRPagedKVCache(
            attn_kind="mha",
            max_batch_size=max_batch_size,
            max_total_seq_len=max_total_seq_len,
            prefill_chunk_size=prefill_chunk_size,
            page_size=page_size,
            support_sliding_window=0,
            layer_partition=relax.ShapeExpr([0, self.num_hidden_layers]),
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            qk_head_dim=self.head_dim,
            v_head_dim=self.head_dim,
            mla_original_qk_head_dim=0,
            mla_original_v_head_dim=0,
            rope_mode=RopeMode.NORMAL,
            rope_scale=1,
            rope_theta=self.rope_theta,
            rope_scaling={},
            rope_ext_factors=relax.PrimValue(0),
            rotary_dim=self.head_dim,
            dtype=self.dtype,
            target=target,
            enable_disaggregation=False,
        )

    def get_default_spec(self):
        mod_spec = {
            "embed": {
                "input_ids": nn.spec.Tensor(["seq_len"], "int32"),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "prefill": {
                "input_embed": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "decode": {
                "input_embed": nn.spec.Tensor([1, 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "create_tir_paged_kv_cache": {
                "max_batch_size": int,
                "max_total_seq_len": int,
                "prefill_chunk_size": int,
                "page_size": int,
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)


######################################################################
# Export the model to Relax IRModule
# ----------------------------------
# After defining the model architecture, we can export the model to the Relax IRModule.
# For demonstration, we only show the part of the model architecture. and parameters.

model_config = QwenConfig()
model = QwenForCasualLM(model_config)
model.to("float16")
mod, named_params = model.export_tvm(spec=model.get_default_spec())
prefill_str = mod["prefill"].script()
print(*prefill_str.split("\n")[3:20], sep="\n")  # Only show the first 10 lines for demonstration
print("        ...")

print("\nParameters:")
pprint(named_params[:])  # Only show the first 5 parameters for demonstration

######################################################################
# Define Optimization Pipeline
# ----------------------------
# We define a series of optimization passes to optimize the model. The optimization pipeline
# is designed specifically for the LLMs.


@register_pipeline("opt_llm")
def _pipeline(  # pylint: disable=too-many-arguments
    ext_mods: List[nn.ExternModule] = None,
):
    ext_mods = ext_mods or []

    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        seq = tvm.transform.Sequential(
            [
                # Phase 1. Passes on high-level operator graph
                # We can enable cublas for further optimization
                relax.transform.FuseTransposeMatmul(),
                # Phase 2. Lowering to TIR, inherited TVM Relax's official "zero" pipeline
                relax.transform.LegalizeOps(),
                relax.transform.AnnotateTIROpPattern(),
                relax.transform.FoldConstant(),
                relax.transform.FuseOps(),
                relax.transform.FuseTIR(),
                # Phase 3. Passes on TIR
                relax.transform.DeadCodeElimination(),
                # Phase 4. Low-level Optimizations
                dlight.ApplyDefaultSchedule(
                    dlight.gpu.Matmul(),
                    dlight.gpu.GEMV(),
                    dlight.gpu.Reduction(),
                    dlight.gpu.GeneralReduction(),
                    dlight.gpu.Fallback(),
                ),
                # Phase 5. Lowering to VM bytecode
                relax.transform.RewriteDataflowReshape(),
                relax.transform.ToNonDataflow(),
                relax.transform.RemovePurityChecking(),
                relax.transform.CallTIRRewrite(),
                relax.transform.StaticPlanBlockMemory(),
                relax.transform.RewriteCUDAGraph(),
                relax.transform.LowerAllocTensor(),
                relax.transform.KillAfterLastUse(),
                relax.transform.LowerRuntimeBuiltin(),
                relax.transform.VMShapeLower(),
                relax.transform.AttachGlobalSymbol(),
                relax.transform.AttachExternModules(ext_mods),
            ]
        )
        mod = seq(mod)
        return mod

    return _pipeline


with target:
    ex = relax.build(mod, target, pipeline=relax.get_pipeline("opt_llm"))
    vm = relax.VirtualMachine(ex, dev)


######################################################################
# Prepare the model weights
# -------------------------
# We load the pre-trained weights from Hugging Face and prepare the model weights.
# The pre-trained weights are stored in the Hugging Face format. We need to load the weights
# and prepare the model parameters.
#
# .. note::
#
#   Note that we won't execute the following code in this tutorial because the pre-trained weights
#   are not available in the CI environment.
#


IS_IN_CI = os.getenv("CI", "") == "true"

# HF_WEIGHT_PATH = None
HF_WEIGHT_PATH = Path("/home/an/Downloads/Qwen2.5-0.5B-Instruct")

if not IS_IN_CI:
    import numpy as np
    import safetensors.torch
    import torch

    if HF_WEIGHT_PATH is None or not HF_WEIGHT_PATH.exists():
        raise ValueError("Please set the HF_WEIGHT_PATH to the path of the pre-trained weights.")

    # Torch format weights
    param_dict = safetensors.torch.load_file(HF_WEIGHT_PATH / "model.safetensors", device="cpu")
    # Numpy format weights
    param_dict = {
        k: v.half().numpy() if v.dtype == torch.bfloat16 else v.numpy()
        for k, v in param_dict.items()
    }


    named_params = dict(named_params)
    print("🔍 Model parameters:")
    for key in param_dict.keys():
        print(key)
    for i in range(model_config.num_hidden_layers):
        # Add QKV in self attention
        attn = f"model.layers.{i}.self_attn"
        param_dict[f"{attn}.qkv_proj.weight"] = np.concatenate(
            [
                param_dict.pop(f"{attn}.q_proj.weight"),  # Pop the old parameters to save memory
                param_dict.pop(f"{attn}.k_proj.weight"),
                param_dict.pop(f"{attn}.v_proj.weight"),
            ],
            axis=0,
        )
        # Add gates in MLP
        mlp = f"model.layers.{i}.mlp"
        param_dict[f"{mlp}.gate_up_proj.weight"] = np.concatenate(
            [
                param_dict.pop(f"{mlp}.gate_proj.weight"),
                param_dict.pop(f"{mlp}.up_proj.weight"),
            ],
            axis=0,
        )
    print("Loaded param_dict keys:", param_dict.keys())
    print("TVM named_params keys:", named_params.keys())
    print("🔍 Checking final param_dict keys before accessing lm_head.weight:")
    for key in param_dict.keys():
        print(key)

    # Convert params into ndarray
    print("🔍 Checking for lm_head.weight in named_params.keys():")
    if "lm_head.weight" in named_params.keys():
        print("✅ lm_head.weight found in named_params")
    else:
        print("❌ lm_head.weight is missing in named_params")
    print("🔍 Checking dtype of lm_head.weight:")
    
    if "lm_head.weight" not in param_dict and "model.embed_tokens.weight" in param_dict:
        param_dict["lm_head.weight"] = param_dict["model.embed_tokens.weight"]
        print("✅ Assigned lm_head.weight from embed_tokens.weight")
    import os
    os.environ["TVM_CUDA_ALLOW_DYNAMIC_SHARED_MEMORY_SIZE"] = "45152"


    params = [
        tvm.nd.array(param_dict[k].astype("float16"), device=dev) for k in named_params.keys()
    ]


    

######################################################################
# Deploy the compiled model
# -------------------------
# After the model and weights are ready, we can deploy the compiled model on the target device.
# The language models inference includes two steps: prefill and decode. The prefill step is
# used to process the input tokens and store the KVCache. The decode step is used to generate
# the token until the end token is generated.


######################################################################
# Tokenization
# ~~~~~~~~~~~~
# The first step is to tokenize the input prompt and embed the tokens into the hidden states.
# The tokenization and embedding are the same as the original model. We use the HF tokenizer
# to tokenize the input prompt and embed the tokens into the hidden states.
# Note that different models require different tokenization and prompt format, please refer to
# the model documentation for the correct tokenization and prompt format.


if not IS_IN_CI:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(HF_WEIGHT_PATH)
    messages = [
        {"role": "user", "content": "What's your name?"},
    ]
    prompt = tokenizer.apply_chat_template(messages)
    input_len = len(prompt)

    # Load prompt tokens into TVM ndarray on the target device
    tokens = tvm.nd.array(np.array(prompt).astype("int32"), device=dev)

######################################################################
# Create the KVCache
# ~~~~~~~~~~~~~~~~~~
# Before starting the inference, we need to create the KVCache. The KVCache is used to store the
# key and value tensors for the attention layer. Apache TVM provides a PagedKVCache to store the
# key and value tensors. We create the PagedKVCache with the specified parameters.

if not IS_IN_CI:
    kv_cache = vm["create_tir_paged_kv_cache"](
        ShapeTuple([1]),  # max_batch_size=1
        ShapeTuple([2048]),  # max_total_seq_len=2048
        ShapeTuple([2048]),  # prefill_chunk_size=2048
        ShapeTuple([16]),  # page_size=16
    )


######################################################################
# Embedding
# ~~~~~~~~~
# The next step is to embed the tokens into the hidden states. We use the `embed` function
# compiled in the Relax IRModule to embed the tokens into the hidden states.

nd_view_func = tvm.get_global_func("vm.builtin.reshape")


def embed(tokens, params):
    _embed = vm["embed"](tokens, params)
    # Reshape hidden from [seq_len, hidden_size] to [1, seq_len, hidden_size]
    _embed = nd_view_func(_embed, ShapeTuple([1, _embed.shape[0], _embed.shape[1]]))
    return _embed


######################################################################
# Prefill
# ~~~~~~~
# Before running the forward pass, we first get some help functions for preparation.

add_sequence_func = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
begin_forward_func = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
end_forward_func = tvm.get_global_func("vm.builtin.kv_state_end_forward")

######################################################################
# As we are creating a new sequence, we need to call `add_sequence_func` to initialize
# the request. Additionally, we need to call `begin_forward_func` to start the forward pass,
# and `end_forward_func` to end the forward pass.

if not IS_IN_CI:
    seq_id = 0
    add_sequence_func(kv_cache, seq_id)
    hidden_states = embed(tokens, params)
    begin_forward_func(kv_cache, ShapeTuple([seq_id]), ShapeTuple([input_len]))
    logits, kv_cache = vm["prefill"](hidden_states, kv_cache, params)
    end_forward_func(kv_cache)

######################################################################
# Now we have the output logits from the prefill step. The logits are used to generate the token
# via sampling. Let's sample the token from the logits.
#
# In this tutorial, we simplify the sampling process and pick the token with the highest
# probability. In practice, we should sample the token based on the probability distribution.
# Also, to make the tutorial concise, we execute the sample process on CPU.


def sample_token(logits):
    logits_np = logits.numpy()
    return np.argmax(logits_np)


if not IS_IN_CI:
    last_token = sample_token(logits)
    output_tokens = [last_token]


######################################################################
# Decode
# ~~~~~~
# After the prefill step, we can start the decode step. The decode step is used to generate the
# token until the end token is generated. We use the `decode` function compiled in the Relax
# IRModule to generate the token.

if not IS_IN_CI:
    print("The generated token:")

    while last_token != tokenizer.eos_token_id:
        tokens = tvm.nd.array(np.array([last_token]).astype("int32"), device=dev)
        hidden_states = embed(tokens, params)
        begin_forward_func(kv_cache, ShapeTuple([seq_id]), ShapeTuple([1]))
        logits, kv_cache = vm["decode"](hidden_states, kv_cache, params)

        end_forward_func(kv_cache)
        last_token = sample_token(logits)
        output_tokens.append(last_token)

    print(tokenizer.decode(output_tokens))
