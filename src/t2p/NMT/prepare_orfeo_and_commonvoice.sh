#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=40000

tmp=tmp
final=final
src=fr
tgt=frp
lang=fr-frp

mkdir -p $tmp
mkdir -p $final

# echo "pre-processing train and valid data..."
# for l in $src $tgt; do
#     cat train.$l | perl $TOKENIZER -threads 8 -a -l $src >> $tmp/train.$l
#     cat valid.$l | perl $TOKENIZER -threads 8 -a -l $src >> $tmp/valid.$l
# done

# echo "pre-processing test data..."
# for l in $src $tgt; do
#     cat test.$l | perl $TOKENIZER -threads 8 -a -l $src > $tmp/test.$l
# done

TRAIN=train.$lang
BPE_CODE=code
rm -f $TRAIN
for l in $src $tgt; do
    cat train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $f > $tmp/bpe.$f
    done
done

perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $final/train 1 250
perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $final/valid 1 250

for L in $src $tgt; do
    cp $tmp/bpe.test.$L $final/test.$L
done


