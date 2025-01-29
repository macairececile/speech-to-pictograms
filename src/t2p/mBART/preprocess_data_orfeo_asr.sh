#!/bin/bash

SRC=fr
TGT=frp

DATA=/gpfswork/rech/czj/uef37or/Fine_tuning_mBART_MT/orfeo_data/asr

DICT=/gpfsstore/rech/czj/uef37or/pretrained_models/mbart.cc25.v2/dict.txt

for a in "whisper_large-v3_test_" "seamless_large-v2_test_"; do
	for f in "cfpb" "cfpp" "clapi" "coralrom" "crfp" "fleuron" "frenchoralnarrative" "ofrom" "reunions" "tcof" "tufs" "valibel" "orfeo"; do
		DEST=$DATA/$a$f
    		fairseq-preprocess \
      			--source-lang ${SRC} \
      			--target-lang ${TGT} \
  			--testpref ${DATA}/$a$f.spm \
  			--destdir ${DEST}/ \
  			--thresholdtgt 0 \
  			--thresholdsrc 0 \
  			--srcdict ${DICT} \
  			--tgtdict ${DICT} \
  			--workers 70
  	done
done

fairseq-preprocess \
	--source-lang ${SRC} \
      	--target-lang ${TGT} \
  	--testpref /gpfswork/rech/czj/uef37or/Fine_tuning_mBART_MT/orfeo_data/asr/test_wav2vec2_orfeo.spm \
  	--destdir /gpfswork/rech/czj/uef37or/Fine_tuning_mBART_MT/orfeo_data/asr/test_wav2vec2_orfeo/ \
  	--thresholdtgt 0 \
  	--thresholdsrc 0 \
  	--srcdict ${DICT} \
  	--tgtdict ${DICT} \
  	--workers 70
