#!/usr/bin/bash

#SBATCH --job-name=nmt_fr_frp_cv_eval
#SBATCH -A czj@v100
#SBATCH -C v100-32g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=05:00:00
#SBATCH --output=nmt_fr_frp_cv_eval
   
fairseq-generate exp_commonvoice/data-bin/commonvoice.tokenized.fr-frp \
    --path exp_commonvoice/checkpoints/nmt_fr_frp_commonvoice/checkpoint.best_bleu_86.0600.pt \
    --batch-size 128 --beam 5 --remove-bpe > gen_cv.out

# grep ^S gen_cv.out | cut -f2- > gen_cv.out.src    
# grep ^H gen_cv.out | cut -f3- > gen_cv.out.sys
# grep ^T gen_cv.out | cut -f2- > gen_cv.out.ref
    
