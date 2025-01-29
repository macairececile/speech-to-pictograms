#!/usr/bin/bash

#SBATCH --job-name=commonvoice_asr_wav2vec2_test
#SBATCH -A czj@v100
#SBATCH -C v100-32g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=05:00:00
#SBATCH --output=commonvoice_asr_wav2vec2_test

python fine_tuning_wav2vec2.py hparams/train_commonvoice.yaml
