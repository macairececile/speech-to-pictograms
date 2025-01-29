#!/bin/bash

SRC=fr
TGT=frp

DATA=/gpfswork/rech/czj/uef37or/Fine_tuning_mBART_MT/propicto_eval/asr_whisper

DICT=/gpfsstore/rech/czj/uef37or/pretrained_models/mbart.cc25.v2/dict.txt

fairseq-preprocess \
	--source-lang ${SRC} \
      	--target-lang ${TGT} \
  	--testpref /gpfswork/rech/czj/uef37or/Fine_tuning_mBART_MT/propicto_eval/asr_whisper/propicto_eval.spm \
  	--destdir /gpfswork/rech/czj/uef37or/Fine_tuning_mBART_MT/propicto_eval/asr_whisper/final/ \
  	--thresholdtgt 0 \
  	--thresholdsrc 0 \
  	--srcdict ${DICT} \
  	--tgtdict ${DICT} \
  	--workers 70
