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
```

### Step 2: run the container

```bash
docker run -it --gpus all  -v /home/ubuntu/marlin_ae/docker_test/marlin_artifact/result:/projects/result --name <your_container_name> marlin_container
```

### Step 3: run MARLIN tests

```
conda activate marlin
python3 test.py
python3 test_2_4.py
```

### Step 4: run kernel benchmarks

```
./runme.sh
python plot.py
```

# Step-by-Step Instructions

(1) To reproduce the results on Fig. 1 and Fig. 12 and Fig. 9

```
./runme.sh
python plot.py
```

(3) To reproduce the results on Fig. 10

Stop the docker container, if running.

```
sudo nvidia-smi --lock-gpu-clocks=885,885 # BASE_GPU_CLOCK
sudo nvidia-smi --lock-memory-clocks=6251,6251 # BASE_MEM_CLOCK
```

```bash
docker run -it --gpus all  -v /home/ubuntu/marlin_ae/docker_test/marlin_artifact/result:/projects/result --name <your_container_name> marlin_container
```

```
./runme.sh
python plot_sustained.py
```