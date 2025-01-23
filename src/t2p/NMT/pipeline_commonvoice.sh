#!/usr/bin/bash

#SBATCH --job-name=nmt_fr_frp_cv
#SBATCH -A czj@v100
#SBATCH -C v100-32g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=20:00:00
#SBATCH --output=nmt_fr_frp_cv

# Binarize the dataset
TEXT=exp_commonvoice/commonvoice_process/final
fairseq-preprocess \
    --source-lang fr --target-lang frp \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/commonvoice.tokenized.fr-frp \
    --workers 20
    
mkdir -p exp_commonvoice/checkpoints/nmt_fr_frp_commonvoice
CUDA_VISIBLE_DEVICES=0 fairseq-train \
    data-bin/commonvoice.tokenized.fr-frp \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --save-dir exp_commonvoice/checkpoints/nmt_fr_frp_commonvoice \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --max-epoch 80 \
    --keep-best-checkpoints 5 \
    --keep-last-epochs 5
    
