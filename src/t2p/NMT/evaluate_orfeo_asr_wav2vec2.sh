#!/usr/bin/bash

#SBATCH --job-name=nmt_fr_frp_orfeo_eval_asr_w2v
#SBATCH -A czj@v100
#SBATCH -C v100-32g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=01:00:00
#SBATCH --output=nmt_fr_frp_orfeo_eval_asr_w2v

fairseq-generate exp_orfeo/data-bin/orfeo.tokenized.fr-frp/test_wav2vec2_orfeo \
    --path exp_orfeo/checkpoints/nmt_fr_frp_orfeo/checkpoint.best_bleu_87.2803.pt \
    --batch-size 128 --beam 5 --remove-bpe > orfeo_wav2vec2.txt
    
