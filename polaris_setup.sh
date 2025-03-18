#!/bin/bash

module load conda/2024-04-29
module load cudatoolkit-standalone/12.4.1

conda create -n unsloth

conda activate unsloth python=3.11 -y
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

cd ~/unsloth
pip install -e .[cu124-ampere-torch251]
pip install vllm
pip install wandb