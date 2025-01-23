#!/usr/bin/bash

#SBATCH --job-name=nmt_fr_frp_cv_eval_asr_seamless
#SBATCH -A czj@v100
#SBATCH -C v100-32g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=01:00:00
#SBATCH --output=nmt_fr_frp_cv_eval_asr_seamless
   
fairseq-generate exp_commonvoice/data-bin/commonvoice.tokenized.fr-frp/seamless_large-v2_test_commonvoice \
    --path exp_commonvoice/checkpoints/nmt_fr_frp_commonvoice/checkpoint.best_bleu_86.0600.pt \
    --batch-size 128 --beam 5 --remove-bpe
    
