#!/bin/bash

SPM=/gpfswork/rech/czj/uef37or/sentencepiece/build/src/spm_encode
MODEL=/gpfsstore/rech/czj/uef37or/pretrained_models/mbart.cc25.v2/sentence.bpe.model

SRC=fr
TGT=frp

DATA=/gpfswork/rech/czj/uef37or/Fine_tuning_mBART_MT/propicto_eval/asr_whisper
OUT=/gpfswork/rech/czj/uef37or/Fine_tuning_mBART_MT/propicto_eval/asr_whisper

${SPM} --model=${MODEL} < ${DATA}/propicto_eval.${SRC} > ${OUT}/propicto_eval.spm.${SRC}
${SPM} --model=${MODEL} < ${DATA}/propicto_eval.${TGT} > ${OUT}/propicto_eval.spm.${TGT}
