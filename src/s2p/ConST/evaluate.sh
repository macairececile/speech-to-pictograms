#!/bin/bash

#SBATCH --job-name=eval
#SBATCH -A czj@v100
#SBATCH -C v100-32g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=20:00:00
#SBATCH --output=eval

cd /gpfswork/rech/czj/uef37or/ConST/scripts/

echo "ORFEO"

fairseq-generate /gpfswork/rech/czj/uef37or/ConST/orfeo_process --gen-subset test_s2p_orfeo_st --task speech_to_text --prefix-size 1 \
--max-tokens 4000000 --max-source-positions 4000000 --beam 10 \
--config-yaml config_st.yaml  --path /gpfswork/rech/czj/uef37or/ConST/checkpoints/orfeo/average_ckpt.pt \
--scoring sacrebleu > const_orfeo.txt

# echo "COMMONVOICE"

# fairseq-generate /gpfswork/rech/czj/uef37or/ConST/commonvoice_process --gen-subset test_commonvoice_s2p_st --task speech_to_text --prefix-size 1 \
# --max-tokens 4000000 --max-source-positions 4000000 --beam 10 \
# --config-yaml config_st.yaml  --path /gpfswork/rech/czj/uef37or/ConST/checkpoints/commonvoice/average_ckpt.pt \
# --scoring sacrebleu > const_commonvoice.txt
