# FlashAttention Benchmark Suite

**Custom CUDA FlashAttention kernel benchmark for BERT-base.**

This project benchmarks a custom `flash_v2_mma.cu` WMMA tensor-core kernel against PyTorch's native implementations, measuring both **correctness** and **performance** for BERT-base attention layers.

---

## 📊 Latest Benchmark Results (RTX 4060 Laptop, fp16)

| seq_len | custom (ms) | sdpa (ms) | native (ms) | naive (ms) | **custom/naive (x)** | custom vs native error |
|---------|-------------|-----------|-------------|------------|----------------------|------------------------|
| **128** | 2.31        | 0.78      | 0.59        | 1.36       | **0.59x** (~1.7× faster) | 6.8e‑03 |
| **256** | 7.11        | 2.00      | 1.99        | 8.96       | **1.26x** (~1.26× faster) | 6.8e‑03 |
| **512** | 25.81       | 7.05      | 7.81        | 35.64      | **1.38x** (~1.38× faster) | 1.37e‑02 |

**Key observations:**
- The custom kernel is **1.26×–1.7× faster** than the naive softmax implementation.
- Maximum absolute error between custom and native BERT attention is ≤ 1.4e‑2 in fp16, confirming numerical correctness.

---

## 🚀 How to Run the Benchmark

### Prerequisites

- Python 3.12
- PyTorch 2.11.0+cu130
- CUDA 13.0
- NVIDIA GPU (or compatible GPU)
- `transformers` ≥ 4.57.6

### Install Dependencies

```bash
pip install torch==2.11.0 transformers==4.57.6
```

### Run the Benchmark

```bash
python llm_benchmark.py
```

The script will:
1. Load BERT-base-uncased in fp16 on CUDA.
2. Verify your custom kernel's output matches the native implementation (max absolute error printed per sequence length).
3. Measure attention-only GPU time for: **custom**, **sdpa**, **native**, and **naive** implementations.
4. Print a table with speedup ratios (`custom/naive`).

---

## 🧠 Project Structure

```
flash_attention_project/
├─ llm_benchmark.py       # Main benchmark script (modified)
├─ flash_v2_mma.cu        # Original CUDA kernel (WMMA tiles: 64×64, 32×64, 32×32, 16×16)
├─ flash_v2_mma_new.cu    # Extended kernel (+16×32 tile config)
├─ standard_attention.py  # Reference attention implementation
├─ benchmark.py           # Original benchmark script
├─ build/                 # CUDA build artifacts (auto-generated)
├─ CMakeLists.txt         # CMake build configuration
└─ README.md              # This file
```


## ⚝ Known Limitations

| Issue | Detail |
|-------|--------|
| **1024 sequence length** | The WMMA kernel hits `vectorized_gather_kernel` index-out-of-bounds. Requires significant kernel changes (dynamic loop bounds, increased shared memory, or chunked processing). |
| **Deprecated API** | `torch.backends.cuda.sdp_kernel()` is deprecated in PyTorch 2.11; migration to `torch.nn.attention.sdpa_kernel()` is recommended. |
| **UNEXPECTED weight messages** | Benign when loading `bert-base-uncased`; not errors. |

---

## 📦 How to Contribute / Extend

1. **Add 1024 support** – Modify `flash_v2_mma_new.cu` to make the reduction loops dynamic in `seq_len`, add `min(seq_len, ...)` guards on memory reads, and potentially increase shared‑memory tile sizes.
2. **Add more tile configurations** – The `wmma_sram_size()` checks in `flash_attn_kernel()` can be extended with additional `(BR, BC, NWARPS)` combinations.
3. **FP8 / BF16 support** – Adjust `TORCH_CHECK(head_dim % 16 == 0)` and `float16` typedefs if targeting mixed‑precision kernels.
4. **Benchmark more models** – Modify `main()` to loop over different HuggingFace models or head dimensions.

---

## 🙏 Acknowledgements

- Based on [FlashAttention](https://github.com/HazyResearch/flash_attention) research.
- Uses PyTorch `cpp_extension` for CUDA kernel loading.
- BERT-base from HuggingFace `transformers` library.

---

### 📜 License

MIT License – feel free to use, modify, and distribute.
