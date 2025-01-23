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

for f in "cfpb" "cfpp" "clapi" "coralrom" "crfp" "fleuron" "frenchoralnarrative" "ofrom" "reunions" "tcof" "tufs" "valibel"; do
    python $BPEROOT/apply_bpe.py -c $BPE_CODE < seamless_large-v2_test_$f.fr > $tmp/bpe.seamless_large-v2_test_$f.$src
done

python $BPEROOT/apply_bpe.py -c $BPE_CODE < seamless_large-v2_test_orfeo.fr > $tmp/bpe.seamless_large-v2_test_orfeo.$src

cd /gpfswork/rech/czj/uef37or/NMT_transformers/

# mkdir -p exp_orfeo/orfeo_process/final/asr/

cd /gpfswork/rech/czj/uef37or/NMT_transformers/exp_orfeo/orfeo_process/

for f in "cfpb" "cfpp" "clapi" "coralrom" "crfp" "fleuron" "frenchoralnarrative" "ofrom" "reunions" "tcof" "tufs" "valibel"; do
    cp $tmp/bpe.seamless_large-v2_test_$f.fr $final/asr/seamless_large-v2_test_$f.fr
    cp $final/$f.frp $final/asr/seamless_large-v2_test_$f.frp
done

cp $tmp/bpe.seamless_large-v2_test_orfeo.fr $final/asr/seamless_large-v2_test_orfeo.fr
cp $final/test.frp $final/asr/seamless_large-v2_test_orfeo.frp

cd /gpfswork/rech/czj/uef37or/NMT_transformers/

TEXT=exp_orfeo/orfeo_process/final/asr
out=seamless

for f in "cfpb" "cfpp" "clapi" "coralrom" "crfp" "fleuron" "frenchoralnarrative" "ofrom" "reunions" "tcof" "tufs" "valibel"; do
    fairseq-preprocess \
        --source-lang fr --target-lang frp \
        --tgtdict exp_orfeo/data-bin/orfeo.tokenized.fr-frp/dict.frp.txt \
        --srcdict exp_orfeo/data-bin/orfeo.tokenized.fr-frp/dict.fr.txt \
        --testpref $TEXT/seamless_large-v2_test_$f \
        --destdir exp_orfeo/data-bin/orfeo.tokenized.fr-frp/$f$out/ \
        --workers 20
done

fairseq-preprocess \
    --source-lang fr --target-lang frp \
    --tgtdict exp_orfeo/data-bin/orfeo.tokenized.fr-frp/dict.frp.txt \
    --srcdict exp_orfeo/data-bin/orfeo.tokenized.fr-frp/dict.fr.txt \
    --testpref $TEXT/seamless_large-v2_test_orfeo \
    --destdir exp_orfeo/data-bin/orfeo.tokenized.fr-frp/orfeo$out/ \
    --workers 20



