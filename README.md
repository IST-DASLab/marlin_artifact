# MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models
This is Marlin, a Mixed Auto-Regressive Linear kernel (and the name of one of the planet's fastest fish), an extremely optimized FP16xINT4 matmul kernel aimed at LLM inference that can deliver close to ideal (4x) speedups up to batchsizes of 16-32 tokens (in contrast to the 1-2 tokens of prior work with comparable speedup).

Additionally, it includes Sparse-Marlin, an extension of the MARLIN kernels adding support to 2:4 weight sparsity, achieving 5.3x speedups on NVIDIA GPUs (Ampere/Ada).

## Requirements:

* NVIDIA GPU with compute capability >= 8.0 (Ampere or Ada, MARLIN is not yet optimized for Hopper)

## Getting Started Guide

### Step 1: build the container from scratch

```bash
cd marlin
docker build -t marlin_container .
docker run -it --gpus all --name <your_container_name> marlin_container
```

### Step 2: run tests

```
python3 test.py
python3 test_2_4.py
```

### Step 3: run kernel benchmarks

```
python3 bench.py
```

# Step-by-Step Instructions

(1) To reproduce the results on Fig 1

```
...
```