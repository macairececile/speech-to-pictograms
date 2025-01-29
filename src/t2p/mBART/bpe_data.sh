#!/bin/bash

SPM=/gpfswork/rech/czj/uef37or/sentencepiece/build/src/spm_encode
MODEL=/gpfsstore/rech/czj/uef37or/pretrained_models/mbart.cc25.v2/sentence.bpe.model

SRC=fr
TGT=frp

TRAIN=train
VALID=valid
TEST=test

# orfeo
# DATA=/gpfswork/rech/czj/uef37or/Fine_tuning_mBART_MT/orfeo_data/

# commonvoice
DATA=/gpfswork/rech/czj/uef37or/Fine_tuning_mBART_MT/commonvoice_data/

${SPM} --model=${MODEL} < ${DATA}/${TRAIN}.${SRC} > ${DATA}/${TRAIN}.spm.${SRC} &
${SPM} --model=${MODEL} < ${DATA}/${TRAIN}.${TGT} > ${DATA}/${TRAIN}.spm.${TGT} &
${SPM} --model=${MODEL} < ${DATA}/${VALID}.${SRC} > ${DATA}/${VALID}.spm.${SRC} &
${SPM} --model=${MODEL} < ${DATA}/${VALID}.${TGT} > ${DATA}/${VALID}.spm.${TGT} &
${SPM} --model=${MODEL} < ${DATA}/${TEST}.${SRC} > ${DATA}/${TEST}.spm.${SRC} &
${SPM} --model=${MODEL} < ${DATA}/${TEST}.${TGT} > ${DATA}/${TEST}.spm.${TGT} &
