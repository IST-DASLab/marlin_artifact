#! /bin/bash

eval "$(conda shell.bash hook)"

conda activate marlin

python test.py

python test_2_4.py