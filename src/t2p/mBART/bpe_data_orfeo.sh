#!/bin/bash

SPM=/gpfswork/rech/czj/uef37or/sentencepiece/build/src/spm_encode
MODEL=/gpfsstore/rech/czj/uef37or/pretrained_models/mbart.cc25.v2/sentence.bpe.model

SRC=fr
TGT=frp

TEST=test

DATA=/gpfswork/rech/czj/uef37or/Fine_tuning_mBART_MT/orfeo_data/test_splits/


for f in "cfpb" "cfpp" "clapi" "coralrom" "crfp" "fleuron" "frenchoralnarrative" "ofrom" "reunions" "tcof" "tufs" "valibel"; do
    ${SPM} --model=${MODEL} < ${DATA}/$f.${SRC} > ${DATA}/$f.spm.${SRC} &
    ${SPM} --model=${MODEL} < ${DATA}/$f.${TGT} > ${DATA}/$f.spm.${TGT}
done
