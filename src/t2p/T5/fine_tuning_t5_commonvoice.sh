#!/bin/bash

#SBATCH --job-name=mt_t5_fr_frp_commonvoice
#SBATCH -A czj@v100
#SBATCH -C v100-32g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=20:00:00
#SBATCH --output=mt_t5_fr_frp_commonvoice

python fine_tune_t5_huggingface.py
