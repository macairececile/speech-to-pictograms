#!/bin/bash

export PATH=/home/getalp/macairec/anaconda3/bin:$PATH

source activate creole

python /home/getalp/macairec/ASR/ASR_grammar/seamlessM4T/inference_seamlessm4t_large_v2_orfeo.py
