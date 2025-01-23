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

python $BPEROOT/apply_bpe.py -c $BPE_CODE < whisper_large-v3_test_orfeo.fr > $tmp/bpe.whisper_large-v3_test_orfeo.$src

cd /gpfswork/rech/czj/uef37or/NMT_transformers/exp_orfeo/orfeo_process/

cp $tmp/bpe.whisper_large-v3_test_orfeo.fr $final/asr/whisper_large-v3_test_orfeo.fr
cp $final/test.frp $final/asr/whisper_large-v3_test_orfeo.frp

cd /gpfswork/rech/czj/uef37or/NMT_transformers/

TEXT=exp_orfeo/orfeo_process/final/asr
out=asr

fairseq-preprocess \
    --source-lang fr --target-lang frp \
    --tgtdict exp_orfeo/data-bin/orfeo.tokenized.fr-frp/dict.frp.txt \
    --srcdict exp_orfeo/data-bin/orfeo.tokenized.fr-frp/dict.fr.txt \
    --testpref $TEXT/whisper_large-v3_test_orfeo \
    --destdir exp_orfeo/data-bin/orfeo.tokenized.fr-frp/orfeo$out/ \
    --workers 20



