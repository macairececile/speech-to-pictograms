#!/bin/bash

#SBATCH --job-name=mt_t5_fr_frp_orfeo_eval_wav2vec2
#SBATCH -A czj@v100
#SBATCH -C v100-32g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=20:00:00
#SBATCH --output=mt_t5_fr_frp_orfeo_eval_wav2vec2

python evaluate_HF_translation_model_test.py --test "/gpfswork/rech/czj/uef37or/Fine_tuning_t5/data_asr/wav2vec2/test_orfeo.csv" --model "/gpfswork/rech/czj/uef37or/Fine_tuning_t5/checkpoints_orfeo/checkpoint-289240/" --output "/gpfswork/rech/czj/uef37or/Output_models_pictos/Wav2vec2_t5/orfeo.txt"
