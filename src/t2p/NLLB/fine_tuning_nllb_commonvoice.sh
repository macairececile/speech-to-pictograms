#!/bin/bash

#SBATCH --job-name=mt_nllb_fr_frp_commonvoice_3
#SBATCH -A czj@v100
#SBATCH -C v100-32g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=20:00:00
#SBATCH --output=mt_nllb_fr_frp_commonvoice_3

python fine_tune_nllb_huggingface_commonvoice.py
