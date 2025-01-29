#!/usr/bin/bash

#SBATCH --job-name=test
#SBATCH -A czj@v100
#SBATCH -C v100-32g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=20:00:00
#SBATCH --output=test


python predict_test.py hparams/test_orfeo.yaml
