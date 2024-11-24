#! /bin/bash

#DIR = /projects/result
DIR=result
mkdir -p $DIR

eval "$(conda shell.bash hook)"

export TORCH_CUDA_ARCH_LIST="8.6 8.0"

echo "MARLIN"
conda activate marlin
log_file=$DIR/marlin_sustained.csv
python benchmark/marlin_bench.py > $log_file
conda deactivate

echo "AWQ"
conda activate awq
log_file=$DIR/awq_sustained.csv
python benchmark/awq_bench.py > $log_file
conda deactivate

echo "exllamav2"
conda activate exllamav2
log_file=$DIR/exllamav2_sustained.csv
cp benchmark/exllamav2_bench.py baselines/exllamav2
python baselines/exllamav2/exllamav2_bench.py > $log_file
conda deactivate

echo "bitsandbytes"
conda activate bitsandbytes
log_file=$DIR/bitsandbytes_sustained.csv
cp benchmark/bitsandbytes_bench.py baselines/bitsandbytes/
python3.10 baselines/bitsandbytes/bitsandbytes_bench.py > $log_file
conda deactivate

echo "torch-nightly"
conda activate torchao
log_file=$DIR/torchao_sustained.csv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3.10 benchmark/ao_bench.py > $log_file
conda deactivate

conda deactivate
python3.10 plot_sustained.py