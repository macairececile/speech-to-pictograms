#!/usr/bin/bash

#SBATCH --job-name=nmt_propicto_eval_wav2vec
#SBATCH -A czj@v100
#SBATCH -C v100-32g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=00:20:00
#SBATCH --output=nmt_propicto_eval_wav2vec

fairseq-generate /gpfswork/rech/czj/uef37or/NMT_transformers/propicto_eval/asr_wav2vec/data-bin/propicto_eval.tokenized.fr-frp \
    --path exp_orfeo/checkpoints/nmt_fr_frp_orfeo/checkpoint.best_bleu_87.2803.pt \
    --batch-size 128 --beam 5 --remove-bpe
