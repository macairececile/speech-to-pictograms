#!/bin/bash

SRC=fr
TGT=frp

DATA=/gpfswork/rech/czj/uef37or/Fine_tuning_mBART_MT/orfeo_data/test_splits



DICT=/gpfsstore/rech/czj/uef37or/pretrained_models/mbart.cc25.v2/dict.txt
for f in "cfpb" "cfpp" "clapi" "coralrom" "crfp" "fleuron" "frenchoralnarrative" "ofrom" "reunions" "tcof" "tufs" "valibel"; do
    mkdir -p $DATA/$f
    DEST=$DATA/$f
    fairseq-preprocess \
      	--source-lang ${SRC} \
      	--target-lang ${TGT} \
  	--testpref ${DATA}/$f.spm \
  	--destdir ${DEST}/ \
  	--thresholdtgt 0 \
  	--thresholdsrc 0 \
  	--srcdict ${DICT} \
  	--tgtdict ${DICT} \
  	--workers 70
done
