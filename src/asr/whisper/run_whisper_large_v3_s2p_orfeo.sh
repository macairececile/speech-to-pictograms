#!/bin/bash

export PATH=/home/getalp/macairec/anaconda3/bin:$PATH

source activate whisper

cd /home/getalp/macairec/ASR/

python whisper_predictions_s2p_orfeo.py
