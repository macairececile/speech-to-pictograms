#!/usr/bin/bash

# Binarize the dataset
TEXT=exp_orfeo/orfeo_process/final
fairseq-preprocess \
    --source-lang fr --target-lang frp \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/orfeo.tokenized.fr-frp \
    --workers 20
