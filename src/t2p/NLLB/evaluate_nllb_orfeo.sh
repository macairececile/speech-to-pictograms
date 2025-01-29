#!/bin/bash

#SBATCH --job-name=mt_nllb_orfeo_eval
#SBATCH -A czj@v100
#SBATCH -C v100-32g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=20:00:00
#SBATCH --output=mt_nllb_orfeo_eval

python evaluate_HF_translation_model.py --valid "/gpfswork/rech/czj/uef37or/Fine_tuning_t5/data_orfeo/valid.csv" --test "/gpfswork/rech/czj/uef37or/Fine_tuning_t5/data_orfeo/test.csv" --model "/gpfswork/rech/czj/uef37or/MT_Fine_tuning_HF/Fine_tuning_NLLB/checkpoints_commonvoice/checkpoint-659240/"
