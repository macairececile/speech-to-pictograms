#!/bin/bash

#SBATCH --job-name=mt_mbart_fr_frp_orfeo
#SBATCH -A czj@v100
#SBATCH -C v100-32g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=20:00:00
#SBATCH --output=mt_mbart_fr_frp_orfeo

# orfeo
DATA=/gpfswork/rech/czj/uef37or/Fine_tuning_mBART_MT/orfeo_data/data 

PRETRAIN=/gpfsstore/rech/czj/uef37or/pretrained_models/mbart.cc25.v2 # fix if you moved the downloaded checkpoint
bpe=/gpfsstore/rech/czj/uef37or/pretrained_models/mbart.cc25.v2/sentence.bpe.model

# mkdir -p models/checkpoints/mt_mbart_fr_frp_orfeo

fairseq-train $DATA \
  --encoder-normalize-before --decoder-normalize-before \
  --arch mbart_large --layernorm-embedding \
  --task translation_from_pretrained_bart \
  --source-lang fr --target-lang frp \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler polynomial_decay --lr 3e-05 --warmup-updates 2500 --total-num-update 40000 \
  --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
  --max-tokens 1024 --update-freq 2 \
  --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 5 \
  --seed 222 --log-format simple --log-interval 2 \
  --langs fr \
  --ddp-backend legacy_ddp \
  --max-epoch 40 \
  --save-dir models/checkpoints/mt_mbart_fr_frp_orfeo \
  --keep-best-checkpoints 5 \
  --keep-last-epochs 5 \
  --restore-file models/checkpoints/mt_mbart_fr_frp_orfeo/checkpoint_last.pt
