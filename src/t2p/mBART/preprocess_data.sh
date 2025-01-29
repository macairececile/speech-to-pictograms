#!/bin/bash

SRC=fr
TGT=frp

# orfeo
# DATA=/gpfswork/rech/czj/uef37or/Fine_tuning_mBART_MT/orfeo_data

# commonvoice
DATA=/gpfswork/rech/czj/uef37or/Fine_tuning_mBART_MT/commonvoice_data

TRAIN=train
VALID=valid
TEST=test

DEST=$DATA
NAME=data

DICT=/gpfsstore/rech/czj/uef37or/pretrained_models/mbart.cc25.v2/dict.txt
fairseq-preprocess \
  --source-lang ${SRC} \
  --target-lang ${TGT} \
  --trainpref ${DATA}/${TRAIN}.spm \
  --validpref ${DATA}/${VALID}.spm \
  --testpref ${DATA}/${TEST}.spm \
  --destdir ${DEST}/${NAME} \
  --thresholdtgt 0 \
  --thresholdsrc 0 \
  --srcdict ${DICT} \
  --tgtdict ${DICT} \
  --workers 70
