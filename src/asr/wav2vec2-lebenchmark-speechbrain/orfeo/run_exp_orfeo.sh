#!/usr/bin/bash

#SBATCH --job-name=orfeo_asr_wav2vec2_ft
#SBATCH -A czj@v100
##SBATCH -C v100-32g
#SBATCH --partition=gpu_p4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --output=orfeo_asr_wav2vec2_ft

# In the case of launching lot of experiment like this, changing the port is needed (if you spawn on a node with distributed launch already it will crash (port occupied))
port=29500

python -m torch.distributed.launch --nproc_per_node=4 --master_port=$port fine_tuning_wav2vec2.py hparams/train_orfeo.yaml --distributed_launch --distributed_backend='nccl' --find_unused_parameters
