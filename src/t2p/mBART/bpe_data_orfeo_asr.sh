#!/bin/bash

SPM=/gpfswork/rech/czj/uef37or/sentencepiece/build/src/spm_encode
MODEL=/gpfsstore/rech/czj/uef37or/pretrained_models/mbart.cc25.v2/sentence.bpe.model

SRC=fr
TGT=frp

whisper=whisper_large-v3_test_
seamless=seamless_large-v2_test_
wav2vec2=test_wav2vec2_orfeo

DATA=/gpfswork/rech/czj/uef37or/NMT_transformers/exp_orfeo/orfeo_process
OUT=/gpfswork/rech/czj/uef37or/Fine_tuning_mBART_MT/orfeo_data

for f in "cfpb" "cfpp" "clapi" "coralrom" "crfp" "fleuron" "frenchoralnarrative" "ofrom" "reunions" "tcof" "tufs" "valibel"; do
    ${SPM} --model=${MODEL} < ${DATA}/$whisper$f.${SRC} > ${OUT}/$whisper$f.spm.${SRC} &
    ${SPM} --model=${MODEL} < ${DATA}/$seamless$f.${SRC} > ${OUT}/$seamless$f.spm.${SRC} &
done

for f in "whisper_large-v3_test_orfeo" "seamless_large-v2_test_orfeo" "test_wav2vec2_orfeo"; do
    ${SPM} --model=${MODEL} < ${DATA}/$f.${SRC} > ${OUT}/$f.spm.${SRC} &
done
