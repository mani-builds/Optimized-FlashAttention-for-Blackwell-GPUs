#!/usr/bin/env python
"""Benchmark the custom flash_v2_mma CUDA kernel on a real LLM (BERT-base).

Times only the attention computation inside every encoder layer, for
sequence lengths 128 / 256 / 512 / 1024 at a realistic inference batch of 16,
comparing:
    - custom  : flash_v2_mma.cu (WMMA tensor cores, fp16)
    - sdpa    : torch.flash_attention backend (F.scaled_dot_product_attention)
    - native  : the model's own eager attention implementation (via F.scaled_dot_product_attention)
    - naive   : materialized softmax attention

BERT-base is used because it computes FULL (non-causal) attention, which is
exactly what the custom kernel implements. Head dim is 64 (multiple of 16,
<= 128), matching the kernel's constraints. The kernel launches grid (B, H),
so a batch of 16 saturates the GPU (192 blocks) unlike B=1.
"""
import math
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils import cpp_extension

from transformers.models.bert.modeling_bert import (
    BertSelfAttention,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SEQUENCE_LENGTHS = [128, 256, 512]  # 1024 not yet supported; kernel hits vectorized_gather out-of-bounds
MODEL_NAME = "bert-base-uncased"
BATCH = 16  # realistic inference batch; grid = (B, H) needs B large to fill the GPU
REPS = 10
WARMUP = 5

os.makedirs(os.path.join(HERE, "build"), exist_ok=True)

sys.stdout.reconfigure(line_buffering=True)
print(f"torch {torch.__version__} | CUDA {torch.version.cuda} | {torch.cuda.get_device_name(0)}")

# ---- global state shared with the monkey-patched BertSelfAttention.forward ----
STATE = {"method": None, "ev_pairs": [], "capture": None, "rep": 0}

# ---- load the custom CUDA kernel once (reuses build/flash_pytorch_extension.so) ----
kernel = cpp_extension.load(
    name="flash_pytorch_extension",
    sources=[os.path.join(HERE, "flash_v2_mma.cu")],
    extra_cflags=["-O3", "-g"],
    extra_include_paths=[],
    build_directory=os.path.join(HERE, "build"),
    verbose=False,
)


def _reshape_mha(t: torch.Tensor, B: int, Np: int) -> torch.Tensor:
    """[B, H, Nd, d] -> [B, N, H*d] contiguous, the layout BERT consumes."""
    return t.transpose(1, 2).reshape(B, Np, -1).contiguous()


def custom_attention(q, k, v):
    B, H, Nd, d = q.shape
    o = kernel.flash_attn_v2_mma(q.contiguous(), k.contiguous(), v.contiguous())
    return _reshape_mha(o, B, Nd)


def sdpa_attention(q, k, v):
    B, H, Nd, d = q.shape
    with torch.backends.cuda.sdp_kernel(enable_flash=True):
        o = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    return _reshape_mha(o, B, Nd)


def naive_attention(q, k, v):
    B, H, Nd, d = q.shape
    scale = 1.0 / math.sqrt(d)
    s = torch.matmul(q, k.transpose(-2, -1)) * scale
    p = torch.softmax(s, dim=-1)
    o = torch.matmul(p, v)
    return _reshape_mha(o, B, Nd)


METHODS = {"custom": custom_attention, "sdpa": sdpa_attention, "naive": naive_attention}
METHOD_NAMES = ["custom", "sdpa", "native", "naive"]


def native_attention(q, k, v):
    """Wrapper to match the interface of other methods: takes (q, k, v) and returns reshaped output."""
    B, H, Nd, d = q.shape
    # Use F.scaled_dot_product_attention with is_causal=False to emulate "native" eager attention
    with torch.backends.cuda.sdp_kernel(enable_flash=False):
        o = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    return _reshape_mha(o, B, Nd)


_orig_forward = BertSelfAttention.forward


def _patched_forward(self, hidden_states, attention_mask=None, past_key_values=None, **kwargs):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.attention_head_size)

    q = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
    k = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
    v = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)

    method = STATE["method"]
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)

    ev0.record()
    if method == "native":
        attn_output = native_attention(q, k, v)
    else:
        attn_output = METHODS[method](q, k, v)
    ev1.record()

    STATE["ev_pairs"].append((STATE["rep"], ev0, ev1))
    flat = attn_output.reshape(*input_shape, -1).contiguous()
    if STATE["capture"] is not None:
        STATE["capture"]["method"][method] = flat
    return flat, None


def run_capture(model, input_ids, method):
    """Run one forward to capture the last layer's attention output."""
    STATE["method"] = method
    STATE["ev_pairs"] = []
    model(input_ids, attention_mask=None)
    torch.cuda.synchronize()
    return STATE["capture"]["method"][method]


def time_forwards(model, input_ids, method, reps):
    """Mean attention-only GPU time (ms) per forward, measured with CUPTI.

    Runs one real forward to capture per-layer (q, k, v) activations, then
    times each method's attention on those tensors under torch.profiler. Only
    attention kernels execute inside the profiled window, so the summed kernel
    durations divided by `reps` is the attention-only time per forward —
    immune to CPU launch gaps.
    """
    STATE["method"] = "native"
    STATE["ev_pairs"] = []
    captures = []
    orig = BertSelfAttention.forward
    def cap(self, hidden_states, attention_mask=None, past_key_values=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.attention_head_size)
        q = self.query(hidden_states).view(*hidden_shape).transpose(1, 2)
        k = self.key(hidden_states).view(*hidden_shape).transpose(1, 2)
        v = self.value(hidden_states).view(*hidden_shape).transpose(1, 2)
        captures.append((q.contiguous(), k.contiguous(), v.contiguous(), self))
        return orig(self, hidden_states, attention_mask, past_key_values, **kwargs)
    BertSelfAttention.forward = cap
    model(input_ids, attention_mask=None)
    torch.cuda.synchronize()
    BertSelfAttention.forward = orig

    if method == "native":
        calls = [lambda q=q, k=k, v=v: native_attention(q, k, v)
                 for q, k, v, m in captures]
    else:
        calls = [lambda q=q, k=k, v=v: METHODS[method](q, k, v)
                 for q, k, v, m in captures]

    for _ in range(WARMUP):
        for fn in calls:
            fn()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        for _ in range(reps):
            for fn in calls:
                fn()
    torch.cuda.synchronize()

    total_us = sum(e.self_device_time_total for e in prof.key_averages())
    return total_us / 1000.0 / reps


def main():
    from transformers import BertConfig, BertModel

    config = BertConfig.from_pretrained(MODEL_NAME)
    model = BertModel.from_pretrained(MODEL_NAME, config=config)
    model.half().eval().to("cuda")
    for p in model.parameters():
        p.requires_grad_(False)

    BertSelfAttention.forward = _patched_forward

    num_heads = config.num_attention_heads
    head_dim = config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
    print(f"model: {MODEL_NAME} | batch={BATCH} | heads={num_heads} head_dim={head_dim}")

    # correctness: custom vs native per seq length (same input for both)
    for L in SEQUENCE_LENGTHS:
        # Reset BERT's internal positional buffers for the new sequence length
        model.embeddings.position_ids = torch.arange(L, device="cuda").expand((1, -1))
        model.embeddings.token_type_ids = torch.zeros((1, L), device="cuda", dtype=torch.long)

        input_ids = torch.randint(0, 30000, (BATCH, L), device="cuda")
        STATE["capture"] = {"method": {}}
        out_custom = run_capture(model, input_ids, "custom")
        out_native = run_capture(model, input_ids, "native")
        err = (out_custom.float() - out_native.float()).abs().max().item()
        print(f"seq_len {L:>4}  custom vs native max abs err = {err:.3e}")
    STATE["capture"] = None

    print("\n=== Attention-only time (sum over all layers, one forward pass) ===")
    header = f"{'seq_len':>8} | " + " | ".join(f"{m:>10}" for m in METHOD_NAMES) + " | custom/naive (x)"
    print(header)
    print("-" * len(header))

    for L in SEQUENCE_LENGTHS:
        model.embeddings.position_ids = torch.arange(L, device="cuda").expand((1, -1))
        model.embeddings.token_type_ids = torch.zeros((1, L), device="cuda", dtype=torch.long)

        input_ids = torch.randint(0, 30000, (BATCH, L), device="cuda")
        timings = {}
        for method in METHOD_NAMES:
            timings[method] = time_forwards(model, input_ids, method, REPS)

        row = f"{L:>8} | " + " | ".join(f"{timings[m]:>10.4f}" for m in METHOD_NAMES)
        row += f" | {timings['naive'] / timings['custom']:>9.2f}x"
        print(row)

    print("\nDone.")


if __name__ == "__main__":
    main()
