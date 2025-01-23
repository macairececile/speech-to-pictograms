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

cd /gpfswork/rech/czj/uef37or/NMT_transformers/propicto_eval/asr_wav2vec

python $BPEROOT/apply_bpe.py -c $BPE_CODE < propicto_eval.fr > $tmp/bpe.propicto_eval.$src
python $BPEROOT/apply_bpe.py -c $BPE_CODE < propicto_eval.frp > $tmp/bpe.propicto_eval.$tgt

cd /gpfswork/rech/czj/uef37or/NMT_transformers/

cd /gpfswork/rech/czj/uef37or/NMT_transformers/propicto_eval/asr_wav2vec

cp $tmp/bpe.propicto_eval.fr $final/propicto_eval.fr
cp $tmp/bpe.propicto_eval.frp $final/propicto_eval.frp

cd /gpfswork/rech/czj/uef37or/NMT_transformers/

TEXT=propicto_eval/asr_wav2vec/final

fairseq-preprocess \
    --source-lang fr --target-lang frp \
    --tgtdict exp_orfeo/data-bin/orfeo.tokenized.fr-frp/dict.frp.txt \
    --srcdict exp_orfeo/data-bin/orfeo.tokenized.fr-frp/dict.fr.txt \
    --testpref $TEXT/propicto_eval \
    --destdir propicto_eval/asr_wav2vec/data-bin/propicto_eval.tokenized.fr-frp/ \
    --workers 20



