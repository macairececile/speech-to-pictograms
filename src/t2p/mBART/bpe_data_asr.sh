#!/bin/bash

SPM=/gpfswork/rech/czj/uef37or/sentencepiece/build/src/spm_encode
MODEL=/gpfsstore/rech/czj/uef37or/pretrained_models/mbart.cc25.v2/sentence.bpe.model

SRC=fr

TEST=test

DATA=/gpfswork/rech/czj/uef37or/Fine_tuning_mBART_MT/commonvoice_data/
whisper=whisper_large-v3_test_commonvoice
seamless=seamless_large-v2_test_commonvoice
wav2vec2=test_wav2vec2_commonvoice

${SPM} --model=${MODEL} < ${DATA}/$whisper.${SRC} > ${DATA}/$whisper.spm.${SRC} &
${SPM} --model=${MODEL} < ${DATA}/$seamless.${SRC} > ${DATA}/$seamless.spm.${SRC} &
${SPM} --model=${MODEL} < ${DATA}/$wav2vec2.${SRC} > ${DATA}/$wav2vec2.spm.${SRC} &
