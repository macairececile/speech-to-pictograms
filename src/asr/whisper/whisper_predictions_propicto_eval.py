#!/usr/bin/python

import pandas as pd
# import evaluate
from argparse import ArgumentParser, RawTextHelpFormatter
import os
import whisper

# wer_metric = evaluate.load("wer")
# cer_metric = evaluate.load("cer")

punctuations = '!?\":,/.;()[]'
punc_table = str.maketrans({key: None for key in punctuations})


def read_tsv(tsv_file):
    return pd.read_csv(tsv_file, sep='\t')


def process_whisper_output(out_dir, c):
    with open(out_dir + c + '.wav.txt', 'r') as f:
        lines = f.readlines()
        lines = [line.replace('\n', '') for line in lines]
        preds_no_punct = [str(line).translate(punc_table) for line in lines]
        preds_process = [line.replace('-', ' ') for line in preds_no_punct]
        return ' '.join(lines), ' '.join([i.lower() for i in preds_process])


def process_refs(ref):
    ref = ref.replace('- ', ' ').replace(' -', ' ').replace('-', ' ')
    ref = str(ref).translate(punc_table)
    return ref


def save_data_pred_whisper(save_path, path_clips, refs, doc_name, refs_process, preds, preds_process, wers, model_name):
    df = pd.DataFrame.from_dict(
        {'utts': path_clips, 'doc_name':doc_name, 'wer': wers, 'references': refs, 'hypothesis': preds, 'refs_process': refs_process,
         'hyps_process': preds_process})
    csv_file = 'propicto_eval_asr_whisper_' + model_name + '.tsv'
    df.to_csv(save_path + csv_file, index=False, header=True, sep='\t')
    return csv_file


def run_whisper_and_get_prediction(test_set, save_path, model_name, index):
    data = read_tsv(test_set)
    clips = data['clips'].values.tolist()[index:]
    corpus_name = data['doc_name'].values.tolist()[index:]
    model = whisper.load_model(model_name)
    preds, preds_process, refs_process, wers = [], [], [], []
    f = open('/home/getalp/macairec/ASR/ASR_grammar/whisper/'+model_name+"/out_propicto_eval.txt", 'a')
    for i, c in enumerate(clips):
        path_clip = '/home/getalp/macairec/propicto-eval-new/clips/'
        result = model.transcribe(path_clip + c, language='fr')
        f.write(c + '\t' + result["text"]+'\n')
    f.close()


def compute_wer_all(predictions, references):
    wer_score = wer_metric.compute(predictions=predictions, references=references)
    cer_score = cer_metric.compute(predictions=predictions, references=references)
    print("WER on test set : ", wer_score)
    print("CER on test set : ", cer_score)


def asr_whisper_cfpp(args):
    run_whisper_and_get_prediction(args.data, args.output, args.model, int(args.index))


if __name__ == '__main__':
    parser = ArgumentParser(description="Run whisper and get predictions and wer + cer scores",
                            formatter_class=RawTextHelpFormatter)

    parser.add_argument('--data', type=str, required=True,
                        help="")
    parser.add_argument('--output', type=str, required=True,
                        help="")
    parser.add_argument('--model', type=str, required=True, choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help="")
    parser.add_argument('--index', type=str, required=True,
                        help="")
    parser.set_defaults(func=asr_whisper_cfpp)

    args = parser.parse_args()
    args.func(args)
