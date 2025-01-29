#!/bin/bash

SRC=fr
TGT=frp

# orfeo
# DATA=/gpfswork/rech/czj/uef37or/Fine_tuning_mBART_MT/orfeo_data

# commonvoice
DATA=/gpfswork/rech/czj/uef37or/Fine_tuning_mBART_MT/commonvoice_data

TEST=test

DEST=$DATA

whisper=whisper_large-v3_test_commonvoice
seamless=seamless_large-v2_test_commonvoice
wav2vec2=test_wav2vec2_commonvoice

DICT=/gpfsstore/rech/czj/uef37or/pretrained_models/mbart.cc25.v2/dict.txt

for f in "whisper_large-v3_test_commonvoice" "seamless_large-v2_test_commonvoice" "test_wav2vec2_commonvoice"; do
    fairseq-preprocess \
      --source-lang ${SRC} \
      --target-lang ${TGT} \
      --testpref ${DATA}/$f.spm \
      --destdir ${DEST}/$f \
      --thresholdtgt 0 \
      --thresholdsrc 0 \
      --srcdict ${DICT} \
      --tgtdict ${DICT} \
      --workers 70
done
