# MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models
This is Marlin, a Mixed Auto-Regressive Linear kernel (and the name of one of the planet's fastest fish), an extremely optimized FP16xINT4 matmul kernel aimed at LLM inference that can deliver close to ideal (4x) speedups up to batchsizes of 16-32 tokens (in contrast to the 1-2 tokens of prior work with comparable speedup).

Additionally, it includes Sparse-Marlin, an extension of the MARLIN kernels adding support to 2:4 weight sparsity, achieving 5.3x speedups on NVIDIA GPUs (Ampere/Ada).

## Requirements:

* NVIDIA GPU with compute capability >= 8.0 (Ampere or Ada, MARLIN is not yet optimized for Hopper)

# Getting Started Guide
The following bash prompts indicate where to execute each command\
```bash
🖥️ >  #Local Machine
🐳 >  #Docker Container
```

### Step 0: Disable ECC support

If ECC is enabled (e.g., on an A10), then the maximum achievable memory bandwidth will be 10-15% lower
than in the official spec sheet as every memory requests will contain checksum overheads. This can be disabled via

```bash
🖥️ > sudo nvidia-smi -e 0
```

which we do in our A10 benchmarks.

### Step 1: Build the container from scratch

```bash
🖥️ > git clone --recurse-submodules https://github.com/IST-DASLab/marlin_artifact.git
🖥️ > cd marlin_artifact
🖥️ > docker build -t marlin_container . # about 20 minutes
```

### Step 2: Run the container

```bash
🖥️ > docker run --rm -it --gpus all -v $(pwd)/result:/projects/result -v $(pwd)/../models:/projects/models --name marlin marlin_container
```

### Step 3: Run microbenchmarks

```bash
🐳 > ./runme.sh # about 10 minutes
```

The results on Figures 1, 12 and 9 can be found in the ```results/``` folder. Specifically, in figures ```peak_smarlin.pdf```, and ```models.pdf```.

# Additional Step-by-Step Instructions

### (4) [Optional] Run MARLIN tests

```bash
🐳 > ./test/runme.sh
```

### (5) To reproduce the results on Fig. 10

Stop the docker container (only if running)

```bash
🐳 > exit
```

In order to reproduce our "sustainable performance" benchmarks, the GPU clocks need to be locked to their respective base values
using. For instance, in the A10:

```bash
🖥️ > sudo nvidia-smi --lock-gpu-clocks=885,885 #BASE_GPU_CLOCK
```

Rerun the container

```bash
🖥️ > docker rm marlin # only if already exists "docker: Error response from daemon: Conflict. The container name "/marlin" is already in use by container"
🖥️ > docker run -it --gpus all  -v $(pwd)/result:/projects/result --name marlin marlin_container
```

inside the container, rerun the benchmark

```bash
🐳 > ./runme_sustained.sh # Check results on the result/ folder
```

[Optional] To reset the GPU again to the initial configuration

```bash
# stop the container
🐳 > exit
# run on your machine
🖥️ > sudo nvidia-smi --gpu-reset
```

### (6) To reproduce the results on Fig. 11 13, 14, Table 1 ? # Jiale