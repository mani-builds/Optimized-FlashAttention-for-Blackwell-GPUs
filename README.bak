# Project: FlashAttention for LLM Inference Optimization

> This project implements and optimizes FlashAttention for large language model (LLM) inference, targeting both standard and Blackwell GPU architectures. The goal is to achieve significant speedups and memory savings over vanilla attention and previous FlashAttention versions, with benchmarking and integration into PyTorch workflows.

## Goals

- Reimplement FlashAttention-2 from scratch, starting with tiled matmul and naive softmax, then progressing to shared memory tiling and online softmax.
- Integrate custom CUDA kernels as a drop-in replacement for ~torch.nn.functional.scaled_dot_product_attention~.
- Benchmark performance and memory usage against PyTorch baseline and official FlashAttention implementations.
- Explore Blackwell-specific optimizations (TMA loads, warp specialization, FP8 GEMM) for maximum throughput.

## Phases

1. /Phase 1/: Reimplementation of FlashAttention-2 on Ada (RTX 4080)
   - Implement basic tiled matmul and naive softmax.
   - Add shared memory tiling and online softmax.
   - Integrate warp-level reductions, causal masking, and PyTorch custom operator.
   - Benchmark on GPT-2 style models (sequence length 128–2048).
2. /Phase 2/: Blackwell GPU Optimization
   - Rent Blackwell B200 on cloud.
   - Test Phase 1 kernel and add Blackwell-specific features: TMA loads, warp specialization, FP8 GEMM.

## Current Progress

- *Phase 1* is underway:
  - ~naive_softmax.cu~ implements basic tiled matmul and naive softmax.
  - ~flash.cu~ implements FlashAttention-style online softmax with block-wise tiling and shared memory.
  - ~main.cpp~ exposes the CUDA kernel as a PyTorch extension.
  - ~bench.py~ benchmarks the custom kernel against PyTorch's ~scaled_dot_product_attention~ using ~torch.utils.cpp_extension~.
- Initial benchmarks are being run on Ada (RTX 4080) GPUs.
- Integration with PyTorch is functional; outputs are compared for correctness.

## Usage

1. Compile and run the CUDA kernels via PyTorch extension:

   ```python
   minimal_attn = load(
       name='minimal_attn',
       sources=['main.cpp', 'flash.cu'],
       extra_cuda_cflags=['-O3', '-arch=sm_89']
   )
   ```

2. Benchmark against PyTorch baseline:

   ```python
   out = minimal_attn.forward(q, k, v)
   out_ref = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
   print(torch.allclose(out, out_ref, atol=1e-2))
   ```

## Benchmarking

- Profiling is performed using ~torch.autograd.profiler~, Nsight Systems, and PyTorch Profiler.
- Target metrics:
  - 2–3× speedup over vanilla PyTorch attention.
  - 20% less VRAM usage.
  - For Blackwell: 1.8–2.4× speedup over FA-2, >70% of theoretical peak TFLOPS, ~2× lower HBM traffic.

## References

- FlashAttention v1 paper: https://arxiv.org/pdf/2205.14135
- Tri Dao’s FlashAttention-2 blog: https://tridao.me/blog/2023/flash2/
- Starter repo: https://github.com/tspeterkim/flash-attention-minimal

## Next Steps

- Complete Phase 1 milestones and benchmarking.
- Begin Phase 2: port kernels to Blackwell, implement TMA loads and FP8 GEMM.
- Further optimize and validate on Llama-3.1, Mistral, GPT-style models.

## Contact

Open an issue, if there are any questions.
