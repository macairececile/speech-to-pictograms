#!/bin/bash

#SBATCH --job-name=mt_mbart_commonvoice_eval_asr
#SBATCH -A czj@v100
#SBATCH -C v100-32g
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=20:00:00
#SBATCH --output=mt_mbart_commonvoice_eval_asr

model_dir=/gpfswork/rech/czj/uef37or/Fine_tuning_mBART_MT/models/checkpoints/mt_mbart_fr_frp_commonvoice_langs
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

for f in "whisper_large-v3_test_commonvoice" "seamless_large-v2_test_commonvoice" "test_wav2vec2_commonvoice"; do
	fairseq-generate /gpfswork/rech/czj/uef37or/Fine_tuning_mBART_MT/commonvoice_data/$f/ \
  		--path $model_dir/checkpoint_best.pt \
  		--task translation_from_pretrained_bart \
  		--gen-subset test \
  		-t frp -s fr \
  		--bpe 'sentencepiece' --sentencepiece-model /gpfsstore/rech/czj/uef37or/pretrained_models/mbart.cc25.v2/sentence.bpe.model \
  		--sacrebleu \
  		--batch-size 32 --langs $langs
done
