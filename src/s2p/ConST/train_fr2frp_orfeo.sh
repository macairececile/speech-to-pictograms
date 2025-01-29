#!/bin/bash

#SBATCH --job-name=const_orfeo
#SBATCH -A czj@v100
#SBATCH -C v100-32g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=20:00:00
#SBATCH --output=const_orfeo

TGT_LANG=frp
MODEL_DIR=/gpfswork/rech/czj/uef37or/ConST/checkpoints/orfeo

fairseq-train /gpfswork/rech/czj/uef37or/ConST/orfeo_process \
    --task speech_to_text_triplet_with_extra_mt \
    --train-subset train_s2p_orfeo_st --valid-subset valid_s2p_orfeo_st \
    --config-yaml config_st.yaml \
    --langpairs fr-${TGT_LANG} --lang-prefix-tok "<lang:${TGT_LANG}>" \
    --max-audio-positions 600000 --max-source-positions 1024 --max-target-positions 1024 \
    --max-audio-tokens 1000000 --max-text-tokens 2000 --max-tokens 1000000 --max-tokens-valid 2000000 \
    --skip-invalid-size-inputs-valid-test \
    --text-data-sample-ratio 0.25 \
    \
    --arch xstnet_base --w2v2-model-path /gpfsstore/rech/czj/uef37or/pretrained_models/wav2vec2-FR-7K-base/checkpoint_best.pt \
    \
    --optimizer adam --clip-norm 10.0 \
    --lr-scheduler inverse_sqrt --lr 1e-4  --warmup-updates 25000  --weight-decay 0.0 \
    \
    --criterion multi_task_cross_entropy_with_contrastive_with_extra_MT \
    --label-smoothing 0.1 --ignore-prefix-size 1 --report-accuracy \
    --contrastive-weight 1.0 --contrastive-temperature 0.02 --contrastive-seqlen-type none \
    \
    --update-freq 2 --max-update 500000 \
    \
    --no-progress-bar --log-format json --log-interval 100 \
    --save-interval-updates 1000 --save-interval 1 \
    --keep-last-epochs 5 --keep-interval-updates 5 --keep-best-checkpoints 5 \
    --max-epoch 40 \
    --save-dir ${MODEL_DIR} \
    --ddp-backend=no_c10d --fp16 \
    \
    --eval-bleu --eval-bleu-args '{"beam": 4, "prefix_size": 1}' \
    --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples \
    --eval-bleu-bpe sentencepiece --eval-bleu-bpe-path /gpfswork/rech/czj/uef37or/ConST/orfeo_process/spm_unigram10000_st.model \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --restore-file /gpfswork/rech/czj/uef37or/ConST/checkpoints/orfeo/checkpoint_last.pt
