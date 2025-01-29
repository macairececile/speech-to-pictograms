#!/bin/bash

export PATH=/home/getalp/macairec/anaconda3/bin:$PATH

source activate whisper

cd /home/getalp/macairec/ASR/

python whisper_predictions_s2p_commonvoice.py --data '/home/getalp/macairec/data/CommonVoice_s2p/test_commonvoice_s2p.csv' --clips "/home/getalp/macairec/data/CommonVoice/cv-corpus-15.0-2023-09-08/fr/dir_001/clips/" --output '/home/getalp/macairec/ASR/ASR_grammar/S2P/' --index '0'
