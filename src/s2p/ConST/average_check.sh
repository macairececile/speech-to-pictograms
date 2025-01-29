#!/bin/bash

#SBATCH --job-name=av_orfeo
#SBATCH -A czj@v100
#SBATCH -C v100-32g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=20:00:00
#SBATCH --output=av_orfeo

cd /gpfswork/rech/czj/uef37or/ConST/scripts/

# python average_checkpoints.py --inputs /gpfswork/rech/czj/uef37or/ConST/checkpoints/orfeo/ --num-update-checkpoints 5 --output /gpfswork/rech/czj/uef37or/ConST/checkpoints/orfeo/average_ckpt.pt

python average_checkpoints.py --inputs /gpfswork/rech/czj/uef37or/ConST/checkpoints/commonvoice/ --num-update-checkpoints 5 --output /gpfswork/rech/czj/uef37or/ConST/checkpoints/commonvoice/average_ckpt.pt
