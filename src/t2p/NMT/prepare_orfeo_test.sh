#!/bin/bash
SCRIPTS=/gpfswork/rech/czj/uef37or/NMT_transformers/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=/gpfswork/rech/czj/uef37or/NMT_transformers/subword-nmt/subword_nmt
BPE_TOKENS=40000

tmp=../tmp
final=../final
src=fr
tgt=frp
lang=fr-frp

BPE_CODE=../code

cd /gpfswork/rech/czj/uef37or/NMT_transformers/exp_orfeo/orfeo_process/test_splits/

for L in $src $tgt; do
    for f in "cfpb".$L "cfpp".$L "clapi".$L "coralrom".$L "crfp".$L "fleuron".$L "frenchoralnarrative".$L "ofrom".$L "reunions".$L "tcof".$L "tufs".$L "valibel".$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $f > $tmp/bpe.$f
    done
done

for L in $src $tgt; do
    for f in "cfpb" "cfpp" "clapi" "coralrom" "crfp" "fleuron" "frenchoralnarrative" "ofrom" "reunions" "tcof" "tufs" "valibel"; do
    	cp $tmp/bpe.$f.$L $final/$f.$L
    done
done

cd /gpfswork/rech/czj/uef37or/NMT_transformers/
TEXT=exp_orfeo/orfeo_process/final

for f in "cfpb" "cfpp" "clapi" "coralrom" "crfp" "fleuron" "frenchoralnarrative" "ofrom" "reunions" "tcof" "tufs" "valibel"; do
    fairseq-preprocess \
        --source-lang fr --target-lang frp \
        --tgtdict exp_orfeo/data-bin/orfeo.tokenized.fr-frp/dict.frp.txt \
        --srcdict exp_orfeo/data-bin/orfeo.tokenized.fr-frp/dict.fr.txt \
        --testpref $TEXT/$f \
        --destdir exp_orfeo/data-bin/orfeo.tokenized.fr-frp/$f/ \
        --workers 20
done


