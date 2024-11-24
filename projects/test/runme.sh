#! /bin/bash

eval "$(conda shell.bash hook)"

conda activate marlin

python test/test.py

python test/test_2_4.py