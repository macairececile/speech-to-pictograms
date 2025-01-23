#!/bin/bash
SCRIPTS=/gpfswork/rech/czj/uef37or/NMT_transformers/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=/gpfswork/rech/czj/uef37or/NMT_transformers/subword-nmt/subword_nmt
BPE_TOKENS=40000

tmp=tmp
final=final
src=fr
tgt=frp
lang=fr-frp

BPE_CODE=code

cd /gpfswork/rech/czj/uef37or/NMT_transformers/exp_orfeo/orfeo_process/

python $BPEROOT/apply_bpe.py -c $BPE_CODE < test_wav2vec2_orfeo.fr > $tmp/bpe.test_wav2vec2_orfeo.fr

cd /gpfswork/rech/czj/uef37or/NMT_transformers/
cd /gpfswork/rech/czj/uef37or/NMT_transformers/exp_orfeo/orfeo_process/

cp $tmp/bpe.test_wav2vec2_orfeo.fr $final/asr/test_wav2vec2_orfeo.fr
cp $final/test.frp $final/asr/test_wav2vec2_orfeo.frp

cd /gpfswork/rech/czj/uef37or/NMT_transformers/

TEXT=exp_orfeo/orfeo_process/final/asr
f=test_wav2vec2_orfeo

fairseq-preprocess \
    --source-lang fr --target-lang frp \
    --tgtdict exp_orfeo/data-bin/orfeo.tokenized.fr-frp/dict.frp.txt \
    --srcdict exp_orfeo/data-bin/orfeo.tokenized.fr-frp/dict.fr.txt \
    --testpref $TEXT/$f \
    --destdir exp_orfeo/data-bin/orfeo.tokenized.fr-frp/$f/ \
    --workers 20



