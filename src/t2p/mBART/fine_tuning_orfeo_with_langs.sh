#!/bin/bash

#SBATCH --job-name=mt_mbart_fr_frp_orfeo_langs
#SBATCH -A czj@v100
#SBATCH -C v100-32g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=20:00:00
#SBATCH --output=mt_mbart_fr_frp_orfeo_langs

# orfeo
DATA=/gpfswork/rech/czj/uef37or/Fine_tuning_mBART_MT/orfeo_data/data 

PRETRAIN=/gpfsstore/rech/czj/uef37or/pretrained_models/mbart.cc25.v2 # fix if you moved the downloaded checkpoint
bpe=/gpfsstore/rech/czj/uef37or/pretrained_models/mbart.cc25.v2/sentence.bpe.model
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

mkdir -p models/checkpoints/mt_mbart_fr_frp_orfeo_langs

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
  --langs $langs \
  --ddp-backend legacy_ddp \
  --max-epoch 40 \
  --save-dir models/checkpoints/mt_mbart_fr_frp_orfeo_langs \
  --keep-best-checkpoints 5 # \
  # --restore-file models/checkpoints/mt_mbart_fr_frp_orfeo_langs/checkpoint_last.pt
