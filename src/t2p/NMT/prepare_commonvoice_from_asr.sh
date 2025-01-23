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

cd /gpfswork/rech/czj/uef37or/NMT_transformers/exp_commonvoice/commonvoice_process/

python $BPEROOT/apply_bpe.py -c $BPE_CODE < seamless_large-v2_test_commonvoice.fr > $tmp/bpe.seamless_large-v2_test_commonvoice.fr

# cd /gpfswork/rech/czj/uef37or/NMT_transformers/
# mkdir -p exp_commonvoice/commonvoice_process/final/asr/
cd /gpfswork/rech/czj/uef37or/NMT_transformers/exp_commonvoice/commonvoice_process/

cp $tmp/bpe.whisper_large-v3_test_commonvoice.fr $final/asr/seamless_large-v2_test_commonvoice.fr
cp $final/test.frp $final/asr/seamless_large-v2_test_commonvoice.frp

cd /gpfswork/rech/czj/uef37or/NMT_transformers/

TEXT=exp_commonvoice/commonvoice_process/final/asr
f=seamless_large-v2_test_commonvoice

fairseq-preprocess \
    --source-lang fr --target-lang frp \
    --tgtdict exp_commonvoice/data-bin/commonvoice.tokenized.fr-frp/dict.frp.txt \
    --srcdict exp_commonvoice/data-bin/commonvoice.tokenized.fr-frp/dict.fr.txt \
    --testpref $TEXT/$f \
    --destdir exp_commonvoice/data-bin/commonvoice.tokenized.fr-frp/$f/ \
    --workers 20



