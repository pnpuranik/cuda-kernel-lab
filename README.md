# cuda-kernel-lab

Each folder includes:
- `kernel.cu` — CUDA implementation
- `main.cpp` or `main.cu` — Driver/test runner
- `README.md` — Brief on strategy & performance notes

---

## 🔍 Kernels Implemented (So Far)

| Kernel         | Description                             | Optimization Techniques Used                   |
|----------------|-----------------------------------------|------------------------------------------------|
| Vector Add     | Element-wise addition                   | Grid/block tuning                              |
| Matrix Multiply| Naive and shared memory tiled version   | Shared memory, loop unrolling                  |
| Reduction      | Sum reduction using multiple patterns   | Warp shuffle, bank conflict avoidance          |
| Softmax        | Numerically stable softmax for ML       | LogSumExp trick, shared memory, coalescing     |
| LayerNorm      | Mean-variance normalization             | Parallel reduction, memory reuse               |

---

## 🚀 Setup Instructions

### 🔧 Requirements
- CUDA Toolkit (11.x or later)
- A CUDA-capable GPU
- CMake (optional)
- Nsight Compute / nvprof for profiling

### 🛠️ Build & Run (Simple)
```bash
cd cuda-kernel-lab/kernels/vector_add
nvcc -o vector_add main.cu
./vector_add
