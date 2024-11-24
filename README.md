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

### Step 1 [Option 1]: Download an already-built docker image
```bash
🖥️ > wget https://zenodo.org/.../marlin.zip
🖥️ > unzip marlin.zip
🖥️ > cd marlin
🖥️ > docker load -i marlin.tar.gz
```

### Step 1 [Option 2]: Build the container from scratch

```bash
🖥️ > git clone --recurse-submodules https://github.com/IST-DASLab/marlin_artifact.git
🖥️ > cd marlin_artifact
🖥️ > docker build -t marlin_container . # about 20 minutes
```

### Step 2: Run the container

```bash
🖥️ > docker run -it --gpus all -v $(pwd)/result:/projects/result --name marlin marlin_container
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

### (6) Roofline Plot: to reproduce the results on Fig. 11

TODO

# End-to-End Benchmarks

We provide end-to-end benchmarks to evaluate the performance of different large language models (LLMs) using the vLLM framework.

## Before Starting

Download LLM checkpoints. You can download the ones you want to test from Hugging Face and place them in the `models` folder.

We use the following checkpoints for our evaluation.

- [Llama-2-7B](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf),
[Llama-2-7B-Marlin](https://huggingface.co/neuralmagic/llama-2-7b-chat-marlin),
[Llama-2-7B-Marlin-Sparse](https://huggingface.co/nm-testing/Llama-2-7b-pruned2.4-Marlin_24)

- [Llama-2-13B](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf),
[Llama-2-13B-Marlin](https://huggingface.co/robertgshaw2/llama-2-13b-chat-marlin)

- [Yi-34B](https://huggingface.co/NousResearch/Nous-Hermes-2-Yi-34B),
[Yi-34B-Marlin](https://huggingface.co/neuralmagic/Nous-Hermes-2-Yi-34B-marlin)

- [Llama-2-70B](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf),
[Llama-2-70B-Marlin](https://huggingface.co/softmax/Llama-2-70b-chat-hf-marlin)

- [Falcon-180B](https://huggingface.co/tiiuae/falcon-180B-chat),
[Falcon-180B-Marlin](https://huggingface.co/softmax/falcon-180B-chat-marlin)

Run the docker container.

```bash
docker run --rm -it --gpus all -v $(pwd)/models:/projects/models --name marlin marlin_container
```

The following commands should all run inside the docker container.

### Batch Benchmark: to reproduce the results on Fig. 13 and Table 1

Adjust the arguments and run the benchmark with `e2e/batch_bench.py`.

**Example Command**

```bash
/root/miniconda3/envs/vllm/bin/python \
e2e/batch_bench.py \
--model-path="/projects/models/CHECKPOINT_PATH" \
--n-gpus=1 \
--batch-size-list 1 2 4 8 16 32 64 128 \
--n-in-tokens=64 \
--n-out-tokens=64 \
--n-warmup-reps=5 \
--n-reps=10 \
--min-runtime=-1 \
--vllm-gpu-memory-utilization=0.9 \
--vllm-enforce-eager=False
```

**Notes**

- **Replace `CHECKPOINT_PATH`** with the actual path to the model checkpoint you want to test.
- **Adjust `--n-gpus`** to the number of GPUs you want to use.
- **Modify `--batch-size-list`** according to the batch sizes you wish to evaluate.
- If you encounter errors, consider tweaking `--vllm-gpu-memory-utilization` and `--vllm-enforce-eager` to suit your hardware capabilities.

**Argument Descriptions**

To customize the benchmarking process using `batch_bench.py`, you can adjust several command-line arguments as per your requirements.

- **`--model-path`** Specify the path to the model checkpoint folder (in Hugging Face format). Replace `CHECKPOINT_PATH` with the actual directory name of the model checkpoint you downloaded. For example:

  ```bash
  --model-path="/projects/models/meta-llama/Llama-2-7b-chat-hf"
  ```

- **`--n-gpus`** Set the number of GPUs you want to utilize for testing. For instance, to use one GPU:

  ```bash
  --n-gpus=1
  ```

- **`--batch-size-list`** Provide a list of batch sizes you wish to test. Modify this list based on your testing needs. Example:

  ```bash
  --batch-size-list 1 2 4 8 16 32 64 128
  ```

- **`--vllm-gpu-memory-utilization`** Adjust the ratio of GPU memory reserved for vLLM. If you encounter CUDA out-of-memory errors due to temporary tensors, decrease this value. Increase it to reserve more memory for the key-value cache, allowing for larger batch sizes. Example:

  ```bash
  --vllm-gpu-memory-utilization=0.9
  ```

- **`--vllm-enforce-eager`** Decide whether to force vLLM to use eager mode. Setting it to `False` enables CUDA Graph for better performance. Setting it to `True` disables CUDA Graph, which may save GPU memory but could reduce speed. Example:

  ```bash
  --vllm-enforce-eager=False
  ```

Other options that are not necessary to change:

- **`--n-in-tokens`** Number of input tokens per prompt.
- **`--n-out-tokens`** Number of tokens to generate.
- **`--n-warmup-reps`** Number of warm-up iterations before benchmarking.
- **`--n-reps`** Number of iterations after warm-up.
- **`--min-runtime`** Minimum runtime in seconds after warm-up (set to a negative value to disable this option, set to a non-negative value to disable `--n-reps`).

**Output Metrics**

This script should give you the total time to generate 2nd-64th tokens, measured in seconds.
You can calculate the speed-ups by running this script on different models and then manually do the division.

Check the output (stdout). The metric is in the `mean_time_exclude_first` field of the printed Python dictionary which looks like the following:

```
{'model_path': ..., 'n_gpus': ..., 'batch_size': ..., 'mean_time_exclude_first': ..., ...}
```

## QPS Benchmark: to reproduce the results on Fig. 14

This benchmark requires a server process and a client process.
We recommend use a terminal multiplexer like `screen`, and run the server and client process in different terminals.

To start a `screen` session, run

```bash
SHELL=/bin/bash screen -S vllm
```

Below are some common `screen` usages:

- Open a new terminal: CTRL+A, then press C.

- Switch between terminals: CTRL+A, then press N (next) or P (previous).

- Exit a terminal: CTRL+D.

For more usages, please refer to `screen`'s documentation.

### Run Server Process

**Example Command**

```bash
/root/miniconda3/envs/vllm/bin/python \
-m vllm.entrypoints.openai.api_server \
--host=0.0.0.0 \
--port=8001 \
--model=/projects/models/CHECKPOINT_PATH \
--tensor-parallel-size=1 \
--gpu-memory-utilization=.9 \
--disable-log-requests
```

**Notes**

- **Replace `CHECKPOINT_PATH`** with the actual path to the model checkpoint you want to test.
- **Adjust `--tensor-parallel-size`** to the number of GPUs you want to use.
- If you encounter errors, consider tweaking `--gpu-memory-utilization` to suit your hardware capabilities. You can also optionally add `--enforce-eager` flag to disable CUDA Graph. For more options, please refer to the vLLM documentation.

**Wait**

You can run the client process only after the server has started. Wait for the server to start until you see the output info:

```
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
```



### Run Client Process

**Example Command**

```bash
/root/miniconda3/envs/vllm/bin/python \
e2e/qps_bench.py \
--host=localhost \
--port=8001 \
--model=/projects/models/CHECKPOINT_PATH \
--request-rate=1 \
--num-prompts=128 \
--seed=0
```

**Notes**

- **Replace `CHECKPOINT_PATH`** with the actual path to the model checkpoint you want to test.
- **Modify `--request-rate` and `--num-prompts`** according to the QPS and the testing time you wish to evaluate. number of prompts = request rate (QPS) * testing time (in seconds). We recomment to test for at least 128 seconds.
- Requests are sent in randomized intervals. You may vary the random seed via `--seed`.

**Output Metrics**

This script should give you the Time to First Token (TTFT) and Time per Output Token (TPOT) metrics.
Check the output (stdout)!


